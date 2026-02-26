import csv 
import json 
import random 
import re 
import shutil 
import statistics 
from concurrent .futures import ThreadPoolExecutor ,as_completed 
from pathlib import Path 
from typing import TYPE_CHECKING ,Dict ,List ,Optional ,Tuple 

if TYPE_CHECKING :
    from PIL import Image as PILImage 

try :
    from .config_types import Config 
    from .data_io import image_num ,list_images ,load_truth ,prepare_detector_dataset ,save_truth_csv 
    from .detector_crop import detect_and_crop_with_meta 
    from .evaluation_utils import _round_floats_for_print ,evaluate_predictions ,find_correct_image_ids 
    from .ocr_easy import _easyocr_to_data_dict ,_get_easyocr_reader ,_preprocess_image_for_ocr_cv2 
    from .parser_heuristics import KEYWORDS ,NUTRIENTS ,_prediction_quality_score ,_refine_macros_with_energy_consistency ,calculate_intake ,extract_nutrients_from_image ,normalize_text 
    from .reporting import print_evaluation_summary 
    from .yolo_train_utils import load_detector_model ,train_detector 
except ImportError :
    from config_types import Config 
    from data_io import image_num ,list_images ,load_truth ,prepare_detector_dataset ,save_truth_csv 
    from detector_crop import detect_and_crop_with_meta 
    from evaluation_utils import _round_floats_for_print ,evaluate_predictions ,find_correct_image_ids 
    from ocr_easy import _easyocr_to_data_dict ,_get_easyocr_reader ,_preprocess_image_for_ocr_cv2 
    from parser_heuristics import KEYWORDS ,NUTRIENTS ,_prediction_quality_score ,_refine_macros_with_energy_consistency ,calculate_intake ,extract_nutrients_from_image ,normalize_text 
    from reporting import print_evaluation_summary 
    from yolo_train_utils import load_detector_model ,train_detector 

def run_full_pipeline (cfg ):
    truth =load_truth (cfg .truth_file )
    cfg .work_dir .mkdir (parents =True ,exist_ok =True )
    save_truth_csv (truth ,cfg .work_dir /"truth_clean.csv")
    detector_path :Optional [Path ]=None 
    split_ids_map :Optional [Dict [str ,List [str ]]]=None 
    split_ids_file =cfg .work_dir /"detector_dataset"/"split_ids.json"
    if cfg .mode !="ocr-only"and cfg .detector_weights is not None :
        if not cfg .detector_weights .exists ():
            raise FileNotFoundError (f"Prosledjeni --detector-weights ne postoji: {cfg .detector_weights }")
        detector_path =cfg .detector_weights 
        print (f"Info: koristim postojeci detector model bez treninga: {detector_path }")
        if split_ids_file .exists ():
            try :
                split_ids_map =json .loads (split_ids_file .read_text (encoding ="utf-8"))
                print (
                f"Info: ucitan postojeci split_ids.json (test={len (split_ids_map .get ('test',[]))})."
                )
            except Exception :
                split_ids_map =None 

    if detector_path is None and cfg .mode =="train":
        if not cfg .labels_dir .exists ()or not list (cfg .labels_dir .glob ("*.txt")):
            raise RuntimeError (
            f"Mode=train zahteva labels folder, ali nije pronadjen/popunjen: {cfg .labels_dir }. "
            "Koristi rucne YOLO anotacije u labels/ ili pokreni --mode ocr-only."
            )
        dataset_yaml ,_split_summary =prepare_detector_dataset (cfg ,truth )
        if split_ids_file .exists ():
            split_ids_map =json .loads (split_ids_file .read_text (encoding ="utf-8"))
        detector_path =train_detector (cfg ,dataset_yaml )
    elif detector_path is None and cfg .mode =="auto":
        if cfg .labels_dir .exists ()and list (cfg .labels_dir .glob ("*.txt")):
            dataset_yaml ,_split_summary =prepare_detector_dataset (cfg ,truth )
            if split_ids_file .exists ():
                split_ids_map =json .loads (split_ids_file .read_text (encoding ="utf-8"))
            detector_path =train_detector (cfg ,dataset_yaml )
        else :
            print ("Info: labels nisu dostupne (auto-label je iskljucen), preskacem detector trening i koristim OCR na celoj slici.")
    elif cfg .mode =="ocr-only":
        print ("Info: mode=ocr-only, preskacem detector trening i koristim OCR na celoj slici.")

    from PIL import Image 

    results :List [Dict [str ,object ]]=[]
    images =list_images (cfg .images_dir )
    if split_ids_map and split_ids_map .get ("test"):
        test_ids =set (str (x ).lower ()for x in split_ids_map ["test"])
        images =[p for p in images if p .stem .lower ()in test_ids ]
        print (f"Info: OCR evaluacija radi se nad test splitom ({len (images )} slika).")
    detector_model =load_detector_model (detector_path )if detector_path is not None else None 

    def _prepare_for_ocr (img_path :Path ):
        image_id =img_path .stem .lower ()
        crop =None 
        if detector_model is not None :
            crop ,_ =detect_and_crop_with_meta (img_path ,detector_model ,cfg .conf )
        return {
        "image_id":image_id ,
        "img_path":img_path ,
        "crop":crop ,
        "truth":truth .get (image_id ),
        }

    def _run_ocr_from_prepared (item :Dict [str ,object ]):
        img_path =item ["img_path"]
        image_id =str (item ["image_id"])
        crop =item .get ("crop")
        full_img =Image .open (img_path ).convert ("RGB")

        if crop is not None :
            pred_crop =extract_nutrients_from_image (crop ,lang =cfg .ocr_lang )
            pred_full =extract_nutrients_from_image (full_img ,lang =cfg .ocr_lang )
            score_crop =_prediction_quality_score (pred_crop )
            score_full =_prediction_quality_score (pred_full )
            if score_crop >=score_full :
                pred =pred_crop 
                ocr_source ="crop"
            else :
                pred =pred_full 
                ocr_source ="full_fallback"
        else :
            pred =extract_nutrients_from_image (full_img ,lang =cfg .ocr_lang )
            ocr_source ="full_no_detection"

        pred =_refine_macros_with_energy_consistency (pred )
        intake =calculate_intake (pred ,cfg .grams )

        return {
        "image_id":image_id ,
        "pred":pred ,
        "intake_for_grams":intake ,
        "truth":item .get ("truth"),
        "ocr_source":ocr_source ,
        }


    prepared_items :List [Dict [str ,object ]]=[]
    for img_path in images :
        prepared_items .append (_prepare_for_ocr (img_path ))


    max_workers =2 
    with ThreadPoolExecutor (max_workers =max_workers )as ex :
        futures =[ex .submit (_run_ocr_from_prepared ,item )for item in prepared_items ]
        for fut in as_completed (futures ):
            results .append (fut .result ())

    results .sort (key =lambda row :image_num (str (row .get ("image_id",""))))

    eval_all =evaluate_predictions (results )
    correct_ids_20 =find_correct_image_ids (results ,rel_tol =0.20 )

    correct_ids_3of4_20 :List [str ]=[]
    for row in results :
        pred =row .get ("pred")or {}
        gt =row .get ("truth")or {}
        ok_count =0 
        total_present =0 
        for nutrient in NUTRIENTS :
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

    print_evaluation_summary (results ,eval_all ,correct_ids_20 ,correct_ids_3of4_20 ,_round_floats_for_print )


def main ():
    print ("Info: trenutno_easyocr.py koristi EasyOCR (isti parser/metrike kao trenutno.py).")
    _get_easyocr_reader ()
    print ("Info: EasyOCR inicijalizovan.")
    # Ako hoces od 0 (YOLO trening + OCR): stavi mode = "train" i detector_weights_path = "".
    # Ako hoces sa postojecim modelom: stavi detector_weights_path na .\\outputs\\runs\\nutrition_yolov8n\\weights\\best.pt.
    project_dir =Path (__file__ ).resolve ().parent .parent
    mode ="auto"
    work_dir_name ="outputs"
    detector_weights_path =r".\outputs\runs\nutrition_yolov8n\weights\best.pt"
    cfg =Config (
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
