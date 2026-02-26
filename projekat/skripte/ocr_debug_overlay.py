import json 
import threading 
from concurrent .futures import ThreadPoolExecutor ,as_completed 
from pathlib import Path 
from typing import Dict ,List ,Optional 
import sys 

sys .path .insert (0 ,str (Path (__file__ ).resolve ().parent .parent ))
from project import trenutno_easyocr as base 

DEBUG_PIC_DIR =Path (__file__ ).resolve ().parent /"debugPic"
DEBUG_PIC_KEEP_LAST =60 
_DEBUG_PIC_LOCK =threading .Lock ()
_DEBUG_PIC_COUNTER =0 

DEBUG_KEYWORD_FRAGMENTS ={
"energija":["energ","ene","gija","energy","kcal"],
"masti":["mast","masti","fat","tuky","fett","zsir"],
"ugljeni_hidrati":["uglj","gljen","hidr","rati","hidrat","carb","sachar"],
"proteini":["prot","prote","protei","tein","belan","belanc","bjelan","bjelanc"],
}


def _debug_norm_token (s :str ):
    import re 

    return re .sub (r"[^a-z0-9]+","",str (s or "").lower ())


def _debug_token_matches_any_keyword (token_text :str ):
    n =_debug_norm_token (token_text )
    if not n :
        return False 
    for nutrient in base .NUTRIENTS :
        for kw in base .KEYWORDS .get (nutrient ,[]):
            k =_debug_norm_token (kw )
            if k and (k in n or n in k ):
                return True 
        for frag in DEBUG_KEYWORD_FRAGMENTS .get (nutrient ,[]):
            if frag in n :
                return True 
    return False 


def _debug_token_matches_nutrient (token_text :str ,nutrient :str ):
    n =_debug_norm_token (token_text )
    if not n :
        return False 
    for kw in base .KEYWORDS .get (nutrient ,[]):
        k =_debug_norm_token (kw )
        if k and (k in n or n in k ):
            return True 
    for frag in DEBUG_KEYWORD_FRAGMENTS .get (nutrient ,[]):
        if frag in n :
            return True 
    return False 


def _debug_token_number_values (token_text :str ):
    import re 

    vals :List [float ]=[]
    for m in re .finditer (r"[0-9]{1,4}(?:[.,][0-9]{1,2})?",str (token_text or "")):
        try :
            vals .append (float (m .group (0 ).replace (",",".")))
        except Exception :
            continue 
    return vals 


def _save_debug_parser_overlay (image_obj ,image_id :str ,ocr_source :str ):
    global _DEBUG_PIC_COUNTER 
    try :
        from PIL import ImageDraw 
    except Exception :
        return 
    try :
        data ,_ =base ._easyocr_to_data_dict (image_obj )
        img =base ._preprocess_image_for_ocr_cv2 (image_obj ).convert ("RGB")
    except Exception :
        return 

    draw =ImageDraw .Draw (img )
    token_boxes :List [Dict [str ,object ]]=[]
    n =len (data .get ("text",[]))
    for i in range (n ):
        raw =str (data .get ("text",[""])[i ]or "").strip ()
        if not raw :
            continue 
        try :
            x =int (data .get ("left",[0 ])[i ]or 0 )
            y =int (data .get ("top",[0 ])[i ]or 0 )
            w =int (data .get ("width",[0 ])[i ]or 0 )
            h =int (data .get ("height",[0 ])[i ]or 0 )
        except Exception :
            continue 
        if w <=0 or h <=0 :
            continue 
        token_boxes .append (
        {
        "text":raw ,
        "x":x ,
        "y":y ,
        "w":w ,
        "h":h ,
        "cx":x +(w /2.0 ),
        "cy":y +(h /2.0 ),
        "x2":x +w ,
        }
        )
        if _debug_token_matches_any_keyword (raw ):
            draw .rectangle ([x ,y ,x +w ,y +h ],outline =(255 ,165 ,0 ),width =2 )

    green_boxes =set ()
    for nutrient in base .NUTRIENTS :
        best_idx :Optional [int ]=None 
        best_dx =float ("inf")
        for kw in token_boxes :
            if not _debug_token_matches_nutrient (str (kw ["text"]),nutrient ):
                continue 
            y_tol =max (8.0 ,float (kw ["h"])*1.0 )
            kx2 =float (kw ["x2"])
            kcy =float (kw ["cy"])
            for idx ,tb in enumerate (token_boxes ):
                if float (tb ["x"])<=kx2 :
                    continue 
                if abs (float (tb ["cy"])-kcy )>y_tol :
                    continue 
                if "%"in str (tb ["text"]):
                    continue 
                if not _debug_token_number_values (str (tb ["text"])):
                    continue 
                dx =float (tb ["x"])-kx2 
                if dx <best_dx :
                    best_dx =dx 
                    best_idx =idx 
        if best_idx is not None :
            green_boxes .add (best_idx )

    for idx ,tb in enumerate (token_boxes ):
        if idx not in green_boxes :
            continue 
        x =int (tb ["x"])
        y =int (tb ["y"])
        w =int (tb ["w"])
        h =int (tb ["h"])
        draw .rectangle ([x ,y ,x +w ,y +h ],outline =(0 ,255 ,0 ),width =3 )

    with _DEBUG_PIC_LOCK :
        DEBUG_PIC_DIR .mkdir (parents =True ,exist_ok =True )
        _DEBUG_PIC_COUNTER +=1 
        out_path =DEBUG_PIC_DIR /f"{image_id }_{ocr_source }_{_DEBUG_PIC_COUNTER :04d}.png"
        try :
            img .save (out_path ,quality =95 )
        except Exception :
            return 
        files =sorted (DEBUG_PIC_DIR .glob ("*.png"),key =lambda p :p .stat ().st_mtime )
        for old in files [:-DEBUG_PIC_KEEP_LAST ]:
            try :
                old .unlink ()
            except Exception :
                pass 


def run_full_pipeline (cfg :base .Config ):
    truth =base .load_truth (cfg .truth_file )
    cfg .work_dir .mkdir (parents =True ,exist_ok =True )
    base .save_truth_csv (truth ,cfg .work_dir /"truth_clean.csv")
    detector_path :Optional [Path ]=None 
    split_ids_map :Optional [Dict [str ,List [str ]]]=None 
    split_ids_file =cfg .work_dir /"detector_dataset"/"split_ids.json"

    if cfg .mode !="ocr-only"and cfg .detector_weights is not None :
        if not cfg .detector_weights .exists ():
            print (f"Upozorenje: detector model ne postoji: {cfg .detector_weights }")
            print ("Izaberi sta dalje:")
            print ("  1) FullProcess (YOLO train + detekcija tabele + EasyOCR + parser + evaluacija)")
            print ("  2) OCR + parser sa vec istreniranim rezultatima nije moguc (nema best.pt) -> ocr-only")
            print ("  3) Prekini")
            choice =input ("Unos [1/2/3]: ").strip ()
            if choice =="1":
                cfg .mode ="train"
                cfg .detector_weights =None 
                print ("Info: prebacujem na mode=train (trening iz labels/).")
            elif choice =="2":
                cfg .mode ="ocr-only"
                cfg .detector_weights =None 
                print ("Info: prebacujem na mode=ocr-only.")
            else :
                raise FileNotFoundError (f"Prosledjeni --detector-weights ne postoji: {cfg .detector_weights }")
        else :
            detector_path =cfg .detector_weights 
            print (f"Info: koristim postojeci detector model bez treninga: {detector_path }")
            if split_ids_file .exists ():
                try :
                    split_ids_map =json .loads (split_ids_file .read_text (encoding ="utf-8"))
                    print (f"Info: ucitan postojeci split_ids.json (test={len (split_ids_map .get ('test',[]))}).")
                except Exception :
                    split_ids_map =None 

    if detector_path is None and cfg .mode =="train":
        if not cfg .labels_dir .exists ()or not list (cfg .labels_dir .glob ("*.txt")):
            raise RuntimeError (
            f"Mode=train zahteva labels folder, ali nije pronadjen/popunjen: {cfg .labels_dir }. "
            "Koristi rucne YOLO anotacije u labels/ ili pokreni --mode ocr-only."
            )
        dataset_yaml ,_ =base .prepare_detector_dataset (cfg ,truth )
        if split_ids_file .exists ():
            split_ids_map =json .loads (split_ids_file .read_text (encoding ="utf-8"))
        detector_path =base .train_detector (cfg ,dataset_yaml )
    elif detector_path is None and cfg .mode =="auto":
        if cfg .labels_dir .exists ()and list (cfg .labels_dir .glob ("*.txt")):
            dataset_yaml ,_ =base .prepare_detector_dataset (cfg ,truth )
            if split_ids_file .exists ():
                split_ids_map =json .loads (split_ids_file .read_text (encoding ="utf-8"))
            detector_path =base .train_detector (cfg ,dataset_yaml )
        else :
            print ("Info: labels nisu dostupne, preskacem detector trening i koristim OCR na celoj slici.")
    elif cfg .mode =="ocr-only":
        print ("Info: mode=ocr-only, preskacem detector trening i koristim OCR na celoj slici.")

    from PIL import Image 

    results :List [Dict [str ,object ]]=[]
    images =base .list_images (cfg .images_dir )
    if split_ids_map and split_ids_map .get ("test"):
        test_ids =set (str (x ).lower ()for x in split_ids_map ["test"])
        images =[p for p in images if p .stem .lower ()in test_ids ]
        print (f"Info: OCR evaluacija radi se nad test splitom ({len (images )} slika).")
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
        image_id =str (item ["image_id"])
        crop_meta =dict (item .get ("crop_meta")or {})
        crop =item .get ("crop")
        full_img =Image .open (img_path ).convert ("RGB")

        if crop is not None :
            pred_crop =base .extract_nutrients_from_image (crop ,lang =cfg .ocr_lang )
            pred_full =base .extract_nutrients_from_image (full_img ,lang =cfg .ocr_lang )
            score_crop =base ._prediction_quality_score (pred_crop )
            score_full =base ._prediction_quality_score (pred_full )
            if score_crop >=score_full :
                chosen_img =crop 
                pred =pred_crop 
                ocr_source ="crop"
                alt_score =score_full 
                chosen_score =score_crop 
            else :
                chosen_img =full_img 
                pred =pred_full 
                ocr_source ="full_fallback"
                alt_score =score_crop 
                chosen_score =score_full 
        else :
            chosen_img =full_img 
            pred =base .extract_nutrients_from_image (full_img ,lang =cfg .ocr_lang )
            chosen_score =base ._prediction_quality_score (pred )
            ocr_source ="full_no_detection"
            alt_score =None 

        pred =base ._refine_macros_with_energy_consistency (pred )
        _save_debug_parser_overlay (chosen_img ,image_id =image_id ,ocr_source =ocr_source )
        intake =base .calculate_intake (pred ,cfg .grams )

        return {
        "image_id":image_id ,
        "pred":pred ,
        "intake_for_grams":intake ,
        "truth":item .get ("truth"),
        "ocr_source":ocr_source ,
        "ocr_selected_score":round (float (chosen_score ),3 ),
        "ocr_alternative_score":(None if alt_score is None else round (float (alt_score ),3 )),
        **crop_meta ,
        }

    prepared_items =[_prepare_for_ocr (p )for p in images ]
    with ThreadPoolExecutor (max_workers =2 )as ex :
        futures =[ex .submit (_run_ocr_from_prepared ,item )for item in prepared_items ]
        for fut in as_completed (futures ):
            results .append (fut .result ())

    results .sort (key =lambda row :base .image_num (str (row .get ("image_id",""))))
    last_ten =results [-10 :]
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
    print ("Poslednjih 10 slika (pred vs truth):")

    def _fmt_pred_truth_line (name ,pred_val ,truth_val ):
        if pred_val is None or truth_val is None :
            return f"  {name }={pred_val } | {name }_truth={truth_val } | diff=None | abs_diff=None"
        try :
            diff =float (pred_val )-float (truth_val )
            abs_diff =abs (diff )
            return (
            f"  {name }={pred_val } | {name }_truth={truth_val } | "
            f"diff={diff :+.1f} | abs_diff={abs_diff :.1f}"
            )
        except Exception :
            return f"  {name }={pred_val } | {name }_truth={truth_val } | diff=None | abs_diff=None"

    for row in last_ten :
        image_id =row .get ("image_id")
        pred =row .get ("pred")or {}
        gt =row .get ("truth")or {}
        print (f"{image_id }:")
        print (_fmt_pred_truth_line ("energija",pred .get ("energija"),gt .get ("energija")))
        print (_fmt_pred_truth_line ("masti",pred .get ("masti"),gt .get ("masti")))
        print (_fmt_pred_truth_line ("ugljeni_hidrati",pred .get ("ugljeni_hidrati"),gt .get ("ugljeni_hidrati")))
        print (_fmt_pred_truth_line ("proteini",pred .get ("proteini"),gt .get ("proteini")))
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
    print ("Info: trenutno_slike.py koristi isti pipeline kao trenutno_easyocr.py + debug overlay slike.")
    base ._get_easyocr_reader ()
    print ("Info: EasyOCR inicijalizovan.")
    # Ako hoces od 0 (YOLO trening + OCR): stavi mode = "train" i detector_weights_path = "".
    # Ako hoces sa postojecim modelom: stavi detector_weights_path na .\\outputs\\runs\\nutrition_yolov8n\\weights\\best.pt.
    project_dir =Path (__file__ ).resolve ().parent .parent
    mode ="auto"
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
    mode =mode ,
    ocr_lang ="srp+eng",
    exclude_truth_from_train =False ,
    split_seed =1389 ,
    detector_weights =((project_dir /detector_weights_path ).resolve ()if detector_weights_path .strip ()else None ),
    )
    run_full_pipeline (cfg )


if __name__ =="__main__":
    main ()
