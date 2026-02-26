import json 
from concurrent .futures import ThreadPoolExecutor ,as_completed 
from pathlib import Path 
from typing import Dict ,List ,Optional 
import sys 

sys .path .insert (0 ,str (Path (__file__ ).resolve ().parent .parent ))
from project import trenutno_easyocr as base 


def run_full_pipeline_all200 (cfg :base .Config ):
    truth =base .load_truth (cfg .truth_file )
    cfg .work_dir .mkdir (parents =True ,exist_ok =True )

    detector_path :Optional [Path ]=None 
    if cfg .mode !="ocr-only"and cfg .detector_weights is not None :
        if not cfg .detector_weights .exists ():
            raise FileNotFoundError (f"Detector model ne postoji: {cfg .detector_weights }")
        detector_path =cfg .detector_weights 
        print (f"Info: koristim postojeci detector model bez treninga: {detector_path }")
    elif cfg .mode =="ocr-only":
        print ("Info: mode=ocr-only, preskacem detector i koristim OCR na celoj slici.")
    else :
        raise RuntimeError (
        "trenutno_easyocr_all200.py je wrapper za evaluaciju nad svih 200 sa postojecim modelom. "
        "Postavi detector_weights na validan best.pt ili koristi mode='ocr-only'."
        )

    from PIL import Image 

    results :List [Dict [str ,object ]]=[]
    images =base .list_images (cfg .images_dir )
    print (f"Info: OCR evaluacija radi se nad svim slikama iz {cfg .images_dir } ({len (images )} slika).")
    detector_model =base .load_detector_model (detector_path )if detector_path is not None else None 

    def _prepare_for_ocr (img_path :Path ):
        image_id =img_path .stem .lower ()
        crop_meta :Dict [str ,object ]={"detector_used":False }
        crop =None 
        if detector_model is not None :
            crop ,crop_meta =base .detect_and_crop_with_meta (img_path ,detector_model ,cfg .conf )
        return {
        "image_id":image_id ,
        "img_path":img_path ,
        "crop":crop ,
        "crop_meta":crop_meta ,
        "truth":truth .get (image_id ),
        }

    def _run_ocr_from_prepared (item :Dict [str ,object ]):
        img_path =item ["img_path"]
        crop_meta =dict (item .get ("crop_meta")or {})
        crop =item .get ("crop")
        full_img =Image .open (img_path ).convert ("RGB")

        if crop is not None :
            pred_crop =base .extract_nutrients_from_image (crop ,lang =cfg .ocr_lang )
            pred_full =base .extract_nutrients_from_image (full_img ,lang =cfg .ocr_lang )
            score_crop =base ._prediction_quality_score (pred_crop )
            score_full =base ._prediction_quality_score (pred_full )
            if score_crop >=score_full :
                pred =pred_crop 
                ocr_source ="crop"
                alt_score =score_full 
                chosen_score =score_crop 
            else :
                pred =pred_full 
                ocr_source ="full_fallback"
                alt_score =score_crop 
                chosen_score =score_full 
        else :
            pred =base .extract_nutrients_from_image (full_img ,lang =cfg .ocr_lang )
            chosen_score =base ._prediction_quality_score (pred )
            ocr_source ="full_no_detection"
            alt_score =None 

        pred =base ._refine_macros_with_energy_consistency (pred )
        intake =base .calculate_intake (pred ,cfg .grams )

        return {
        "image_id":str (item ["image_id"]),
        "pred":pred ,
        "intake_for_grams":intake ,
        "truth":item .get ("truth"),
        "ocr_source":ocr_source ,
        "ocr_selected_score":round (float (chosen_score ),3 ),
        "ocr_alternative_score":(None if alt_score is None else round (float (alt_score ),3 )),
        **crop_meta ,
        }

    prepared_items =[_prepare_for_ocr (img_path )for img_path in images ]

    max_workers =2 
    with ThreadPoolExecutor (max_workers =max_workers )as ex :
        futures =[ex .submit (_run_ocr_from_prepared ,item )for item in prepared_items ]
        for fut in as_completed (futures ):
            results .append (fut .result ())

    results .sort (key =lambda row :base .image_num (str (row .get ("image_id",""))))

    eval_all =base .evaluate_predictions (results )
    correct_ids_20 =base .find_correct_image_ids (results ,rel_tol =0.20 )

    correct_ids_3of4_20 :List [str ]=[]
    for row in results :
        pred =row .get ("pred")or {}
        gt =row .get ("truth")or {}
        ok_count =0 
        total_present =0 
        for nutrient in base .NUTRIENTS :
            pv =pred .get (nutrient )
            tv =gt .get (nutrient )
            if pv is None or tv is None :
                continue 
            total_present +=1 
            try :
                tvf =float (tv )
                pvf =float (pv )
            except Exception :
                continue 
            if abs (tvf )<1e-9 :
                if abs (pvf -tvf )<1e-9 :
                    ok_count +=1 
            else :
                if abs (pvf -tvf )/abs (tvf )<=0.20 :
                    ok_count +=1 
        if total_present >=4 and ok_count >=3 :
            correct_ids_3of4_20 .append (str (row .get ("image_id")or ""))

    print ("Zavrseno.")
    print ("")
    print ("Ukupne metrike (sve slike):")
    print (f"processed_images: {len (results )}")
    print (f"samples_with_truth: {eval_all .get ('samples_with_truth',0 )}")

    metric_print_order =[
    "coverage",
    "mae",
    "found_count",
    "sum_error_signed",
    "mean_error_signed",
    "accuracy_within_20_percent",
    ]
    metric_labels ={
    "coverage":"coverage",
    "mae":"mae",
    "found_count":"found_count",
    "sum_error_signed":"sum_error_signed (zbir pred-truth)",
    "mean_error_signed":"mean_error_signed (pred - truth)",
    "accuracy_within_20_percent":"accuracy_within_20_percent",
    }
    for key in metric_print_order :
        if key in eval_all :
            print (f"{metric_labels .get (key ,key )}:")
            print (json .dumps (base ._round_floats_for_print (eval_all [key ],2 ),ensure_ascii =False ,indent =2 ))

    ocr_source_counts :Dict [str ,int ]={}
    for row in results :
        src =str (row .get ("ocr_source")or "unknown")
        ocr_source_counts [src ]=ocr_source_counts .get (src ,0 )+1 
    print ("ocr_source_counts:")
    print (json .dumps (base ._round_floats_for_print (ocr_source_counts ,2 ),ensure_ascii =False ,indent =2 ))
    print ("")
    print (f"Broj slika tacno uradjenih (sva 4 nutrienta unutar +-20%): {len (correct_ids_20 )}")
    print (f"Broj slika tacno uradjenih (3/4 nutrienta unutar +-20%): {len (correct_ids_3of4_20 )}")


def main ():
    print ("Info: trenutno_easyocr_all200.py koristi funkcije iz trenutno_easyocr.py (all200 wrapper).")
    base ._get_easyocr_reader ()
    print ("Info: EasyOCR inicijalizovan.")

    project_dir =Path (__file__ ).resolve ().parent .parent
    work_dir_name ="outputs"
    detector_weights_path =r".\outputs\runs\nutrition_yolov8n\weights\best.pt"

    cfg =base .Config (
    project_dir =project_dir ,
    images_dir =project_dir /"cleanData2",
    labels_dir =project_dir /"labels",
    truth_file =project_dir /"truth.txt",
    work_dir =project_dir /work_dir_name ,
    epochs =20 ,
    imgsz =640 ,
    batch =8 ,
    device ="cpu",
    grams =100.0 ,
    conf =0.25 ,
    run_name ="nutrition_yolov8n",
    mode ="auto",
    ocr_lang ="srp+eng",
    exclude_truth_from_train =False ,
    split_seed =1389 ,
    detector_weights =((project_dir /detector_weights_path ).resolve ()if detector_weights_path .strip ()else None ),
    )
    run_full_pipeline_all200 (cfg )


if __name__ =="__main__":
    main ()
