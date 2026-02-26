import csv 
import json 
import random 
import re 
import shutil 
import statistics 
import threading 
from concurrent .futures import ThreadPoolExecutor ,as_completed 
from dataclasses import dataclass 
from pathlib import Path 
from typing import TYPE_CHECKING ,Dict ,List ,Optional ,Tuple 

if TYPE_CHECKING :
    from PIL import Image as PILImage 


NUTRIENTS =["energija","masti","ugljeni_hidrati","proteini"]
_EASYOCR_INIT_LOCK =threading .Lock ()
_EASYOCR_INFER_LOCK =threading .Lock ()
_EASYOCR_READER =None 

KEYWORDS ={
"energija":["energija","energy","energie","energia"],
"masti":["masti","fat","tuky","zsir","mast","fett"],
"ugljeni_hidrati":[
"ugljeni hidrati",
"ugljeni-hidrati",
"carbohydrate",
"carbohydrates",
"sacharidy",
"szenhidrat",
"uhljohydraty",
],
"proteini":[
"proteini",
"protein",
"bielkoviny",
"feherje",
"proteine",
"belancevine",
"belančevine",
"bjelancevine",
],
}

TABLE_TITLE_HINTS =[
"hranljive vrednosti",
"hranjive vrijednosti",
"nutritivne vrednosti",
"nutritivne vrijednosti",
"nutrition facts",
"nutritional values",
"nutrition",
]

def _get_easyocr_reader ():
    global _EASYOCR_READER 
    if _EASYOCR_READER is not None :
        return _EASYOCR_READER 
    try :
        import easyocr 
    except Exception as exc :
        raise RuntimeError (
        "EasyOCR import nije uspeo (paket mozda nedostaje ili mu zavisnost puca pri importu). "
        f"Originalna greska: {type (exc ).__name__ }: {exc }"
        )from exc 

    with _EASYOCR_INIT_LOCK :
        if _EASYOCR_READER is not None :
            return _EASYOCR_READER 
        last_exc :Optional [Exception ]=None 
        for langs in (["en"],["en","hr"],["en","cs"]):
            try :
                _EASYOCR_READER =easyocr .Reader (langs ,gpu =False ,verbose =False )
                return _EASYOCR_READER 
            except Exception as exc :
                last_exc =exc 
        raise RuntimeError (f"Ne mogu da inicijalizujem EasyOCR: {last_exc }")


def _easy_quad_to_rect (pts :List [List [float ]]):
    xs =[float (p [0 ])for p in pts ]
    ys =[float (p [1 ])for p in pts ]
    x1 =int (max (0.0 ,min (xs )))
    y1 =int (max (0.0 ,min (ys )))
    x2 =int (max (xs ))
    y2 =int (max (ys ))
    return x1 ,y1 ,max (x1 +1 ,x2 ),max (y1 +1 ,y2 )


def _easyocr_lines (image_obj ):
    import numpy as np 

    reader =_get_easyocr_reader ()
    prep =_preprocess_image_for_ocr_cv2 (image_obj )
    arr =np .array (prep .convert ("RGB"))
    out :List [Tuple [List [List [float ]],str ,float ]]=[]

    with _EASYOCR_INFER_LOCK :
        res =reader .readtext (arr ,detail =1 ,paragraph =False )
    for item in res or []:
        try :
            box ,txt ,conf =item [0 ],str (item [1 ]or "").strip (),float (item [2 ]or 0.0 )
        except Exception :
            continue 
        if not txt :
            continue 
        try :
            pts =[[float (p [0 ]),float (p [1 ])]for p in box [:4 ]]
        except Exception :
            continue 
        out .append ((pts ,txt ,conf ))

    if out :
        return out 


    raw_arr =np .array (image_obj .convert ("RGB"))
    with _EASYOCR_INFER_LOCK :
        res2 =reader .readtext (raw_arr ,detail =1 ,paragraph =False )
    for item in res2 or []:
        try :
            box ,txt ,conf =item [0 ],str (item [1 ]or "").strip (),float (item [2 ]or 0.0 )
        except Exception :
            continue 
        if not txt :
            continue 
        try :
            pts =[[float (p [0 ]),float (p [1 ])]for p in box [:4 ]]
        except Exception :
            continue 
        out .append ((pts ,txt ,conf ))
    return out 


def _easyocr_to_data_dict (image_obj ):
    lines =_easyocr_lines (image_obj )
    data :Dict [str ,List [object ]]={
    "text":[],
    "conf":[],
    "left":[],
    "top":[],
    "width":[],
    "height":[],
    "block_num":[],
    "par_num":[],
    "line_num":[],
    }
    rows :List [Tuple [int ,int ,int ,int ,str ,float ]]=[]
    for pts ,txt ,conf in lines :
        x1 ,y1 ,x2 ,y2 =_easy_quad_to_rect (pts )
        rows .append ((x1 ,y1 ,x2 ,y2 ,txt ,conf ))
    rows .sort (key =lambda r :(r [1 ],r [0 ]))

    text_lines :List [str ]=[]
    for idx ,(x1 ,y1 ,x2 ,y2 ,txt ,conf )in enumerate (rows ,start =1 ):
        data ["text"].append (txt )
        data ["conf"].append (round (max (0.0 ,min (100.0 ,conf *100.0 )),2 ))
        data ["left"].append (x1 )
        data ["top"].append (y1 )
        data ["width"].append (max (1 ,x2 -x1 ))
        data ["height"].append (max (1 ,y2 -y1 ))
        data ["block_num"].append (1 )
        data ["par_num"].append (1 )
        data ["line_num"].append (idx )
        text_lines .append (txt )
    return data ,"\n".join (text_lines )


@dataclass 
class Config :
    project_dir :Path 
    images_dir :Path 
    labels_dir :Path 
    truth_file :Path 
    work_dir :Path 
    epochs :int 
    imgsz :int 
    batch :int 
    device :str 
    grams :float 
    conf :float 
    run_name :str 
    mode :str 
    ocr_lang :str 
    exclude_truth_from_train :bool 
    split_seed :int 
    detector_weights :Optional [Path ]


def parse_float (text :str ):
    return float (text .strip ().replace (",","."))


def image_num (image_id :str ):
    return int (re .sub (r"\D","",image_id )or "0")


def parse_truth_line (line :str ):
    line =line .strip ()
    if not line :
        return None 

    m =re .match (r"^\s*(slika\d+|sliak\d+)\s*,\s*(.*)$",line ,flags =re .IGNORECASE )
    if not m :
        return None 
    image_id =m .group (1 ).lower ().replace ("sliak","slika")
    rest =m .group (2 )
    patterns ={
    "energija":r"energija\s*=\s*([0-9]+(?:[\.,][0-9]+)?)",
    "masti":r"masti\s*=\s*([0-9]+(?:[\.,][0-9]+)?)",
    "ugljeni_hidrati":r"ugljeni[-\s]*hidrati\s*=\s*([0-9]+(?:[\.,][0-9]+)?)",
    "proteini":r"proteini\s*=\s*([0-9]+(?:[\.,][0-9]+)?)",
    }
    values :Dict [str ,float ]={}
    for k ,p in patterns .items ():
        pm =re .search (p ,rest ,flags =re .IGNORECASE )
        if not pm :
            return None 
        values [k ]=parse_float (pm .group (1 ))
    return image_id ,values 


def load_truth (path :Path ):
    if not path .exists ():
        raise FileNotFoundError (f"Ne postoji truth fajl: {path }")
    truth :Dict [str ,Dict [str ,float ]]={}
    bad =[]
    for i ,line in enumerate (path .read_text (encoding ="utf-8",errors ="ignore").splitlines (),1 ):
        parsed =parse_truth_line (line )
        if parsed is None :
            if line .strip ():
                bad .append (f"Linija {i }: {line }")
            continue 
        image_id ,values =parsed 
        truth [image_id ]=values 
    if bad :
        print ("Upozorenje: neke truth linije nisu parsirane:")
        for row in bad :
            print (f"  - {row }")
    return truth 


def save_truth_csv (truth :Dict [str ,Dict [str ,float ]],out_csv :Path ):
    out_csv .parent .mkdir (parents =True ,exist_ok =True )
    with out_csv .open ("w",newline ="",encoding ="utf-8")as f :
        w =csv .writer (f )
        w .writerow (["image_id"]+NUTRIENTS )
        for image_id in sorted (truth .keys (),key =image_num ):
            w .writerow ([image_id ]+[truth [image_id ][k ]for k in NUTRIENTS ])


def list_images (images_dir :Path ):
    images =sorted (images_dir .glob ("slika*.jpeg"),key =lambda p :image_num (p .stem ))
    if not images :
        raise RuntimeError (f"Nema slika u: {images_dir }")
    return images 


def split_ids (
ids :List [str ],
train_ratio :float =0.70 ,
val_ratio :float =0.15 ,
seed :int =42 ,
)->Dict [str ,List [str ]]:
    ids =ids .copy ()
    random .Random (seed ).shuffle (ids )
    n =len (ids )
    train_n =int (n *train_ratio )
    val_n =int (n *val_ratio )
    return {
    "train":ids [:train_n ],
    "val":ids [train_n :train_n +val_n ],
    "test":ids [train_n +val_n :],
    }


def prepare_detector_dataset (cfg :Config ,truth :Dict [str ,Dict [str ,float ]]):
    images =list_images (cfg .images_dir )
    print (f"Info: priprema detector skupa koristi {len (images )} slika iz {cfg .images_dir }.")
    ids =[p .stem .lower ()for p in images ]
    truth_ids =set (truth .keys ())
    labeled_ids =[image_id for image_id in ids if (cfg .labels_dir /f"{image_id }.txt").exists ()]
    excluded_truth_ids :List [str ]=[]
    if cfg .exclude_truth_from_train :
        excluded_truth_ids =[image_id for image_id in labeled_ids if image_id in truth_ids ]
        labeled_ids =[image_id for image_id in labeled_ids if image_id not in truth_ids ]
    unlabeled_ids =[image_id for image_id in ids if image_id not in set (labeled_ids )]
    if not labeled_ids :
        raise RuntimeError (
        "Nema nijedne YOLO anotacije za trening. Dodaj labels ili ukljuci auto-label."
        )

    split =split_ids (labeled_ids ,seed =cfg .split_seed )
    ds_root =cfg .work_dir /"detector_dataset"
    if ds_root .exists ():
        shutil .rmtree (ds_root )
    for s in ["train","val","test"]:
        (ds_root /"images"/s ).mkdir (parents =True ,exist_ok =True )
        (ds_root /"labels"/s ).mkdir (parents =True ,exist_ok =True )

    for s ,image_ids in split .items ():
        for image_id in image_ids :
            src_img =cfg .images_dir /f"{image_id }.jpeg"
            src_lbl =cfg .labels_dir /f"{image_id }.txt"
            if not src_img .exists ():
                continue 
            shutil .copy2 (src_img ,ds_root /"images"/s /src_img .name )
            shutil .copy2 (src_lbl ,ds_root /"labels"/s /src_lbl .name )

    (ds_root /"split_ids.json").write_text (
    json .dumps (split ,ensure_ascii =False ,indent =2 ),
    encoding ="utf-8",
    )

    if unlabeled_ids :
        missing_path =cfg .work_dir /"unlabeled_images.txt"
        missing_path .write_text ("\n".join (unlabeled_ids ),encoding ="utf-8")
        print (
        f"Upozorenje: {len (unlabeled_ids )} slika nema labels i nisu ukljucene u detector trening. "
        f"Spisak: {missing_path }"
        )
    if excluded_truth_ids :
        excluded_path =cfg .work_dir /"excluded_from_train_truth_ids.txt"
        excluded_path .write_text ("\n".join (excluded_truth_ids ),encoding ="utf-8")
        print (
        f"Info: {len (excluded_truth_ids )} labelovanih slika iz truth skupa je izbaceno iz YOLO treninga. "
        f"Spisak: {excluded_path }"
        )

    yaml_path =ds_root /"dataset.yaml"
    yaml_path .write_text (
    f"path: {ds_root .as_posix ()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    "  0: nutrition_table\n",
    encoding ="utf-8",
    )
    summary ={k :len (v )for k ,v in split .items ()}
    summary ["total_labeled"]=len (labeled_ids )
    summary ["total_unlabeled"]=len (unlabeled_ids )
    summary ["excluded_truth_from_train"]=len (excluded_truth_ids )
    print (f"Dataset spreman: {summary }")
    return yaml_path ,summary 


def train_detector (cfg :Config ,dataset_yaml :Path ):
    try :
        from ultralytics import YOLO 
    except Exception as exc :
        raise RuntimeError ("Nedostaje ultralytics. Instaliraj: pip install ultralytics")from exc 

    runs_dir =cfg .work_dir /"runs"
    model =YOLO ("yolov8n.pt")
    model .train (
    data =str (dataset_yaml ),
    epochs =cfg .epochs ,
    imgsz =cfg .imgsz ,
    batch =cfg .batch ,
    device =cfg .device ,
    project =str (runs_dir ),
    name =cfg .run_name ,
    pretrained =True ,
    )
    best =runs_dir /cfg .run_name /"weights"/"best.pt"
    if not best .exists ():
        raise RuntimeError (f"Trening zavrsen, ali best.pt nije pronadjen: {best }")
    return best 


def load_detector_model (detector_path :Optional [Path ]):
    try :
        from ultralytics import YOLO 
    except Exception :
        return None 
    if detector_path is None :
        return None 
    return YOLO (str (detector_path ))


def _rank_detection_boxes (result ,img_w :int ,img_h :int ):
    if not hasattr (result ,"boxes")or result .boxes is None or len (result .boxes )==0 :
        return []
    boxes_xyxy =result .boxes .xyxy 
    confs =getattr (result .boxes ,"conf",None )
    img_area =float (max (1 ,img_w *img_h ))
    ranked :List [Tuple [Tuple [int ,int ,int ,int ],float ]]=[]

    for i in range (len (boxes_xyxy )):
        try :
            x1f ,y1f ,x2f ,y2f =boxes_xyxy [i ].tolist ()
        except Exception :
            continue 
        x1 =int (max (0 ,min (img_w -1 ,x1f )))
        y1 =int (max (0 ,min (img_h -1 ,y1f )))
        x2 =int (max (0 ,min (img_w ,x2f )))
        y2 =int (max (0 ,min (img_h ,y2f )))
        bw =x2 -x1 
        bh =y2 -y1 
        if bw <=4 or bh <=4 :
            continue 

        area_ratio =(bw *bh )/img_area 
        conf =0.0 
        if confs is not None :
            try :
                conf =float (confs [i ].item ()if hasattr (confs [i ],"item")else confs [i ])
            except Exception :
                conf =0.0 

        score =conf *8.0 

        if 0.05 <=area_ratio <=0.60 :
            score +=4.0 
        elif 0.02 <=area_ratio <=0.80 :
            score +=1.5 
        if area_ratio >0.85 :
            score -=6.0 
        if area_ratio >0.95 :
            score -=10.0 
        if area_ratio <0.01 :
            score -=4.0 


        aspect =bw /max (1.0 ,float (bh ))
        if 0.6 <=aspect <=4.5 :
            score +=1.0 

        ranked .append (((x1 ,y1 ,x2 ,y2 ),score ))

    ranked .sort (key =lambda item :item [1 ],reverse =True )
    return ranked 


def _expand_box_with_padding (
box :Tuple [int ,int ,int ,int ],
img_w :int ,
img_h :int ,
pad_x_ratio :float =0.16 ,
pad_y_ratio :float =0.22 ,
)->Tuple [int ,int ,int ,int ]:
    x1 ,y1 ,x2 ,y2 =box 
    pad_x =max (8 ,int ((x2 -x1 )*pad_x_ratio ))
    pad_y =max (10 ,int ((y2 -y1 )*pad_y_ratio ))
    return (
    max (0 ,x1 -pad_x ),
    max (0 ,y1 -pad_y ),
    min (img_w ,x2 +pad_x ),
    min (img_h ,y2 +pad_y ),
    )


def _ocr_score_detection_crop (crop_img ):
    try :
        data ,_ =_easyocr_to_data_dict (crop_img )
    except Exception :
        return 0.0 ,None 
    tokens :List [Dict [str ,object ]]=[]
    n =len (data .get ("text",[]))
    for i in range (n ):
        txt =str (data .get ("text",[""])[i ]or "").strip ()
        if not txt :
            continue 
        x =int (data .get ("left",[0 ])[i ]or 0 )
        y =int (data .get ("top",[0 ])[i ]or 0 )
        w =int (data .get ("width",[0 ])[i ]or 0 )
        h =int (data .get ("height",[0 ])[i ]or 0 )
        if w <=0 or h <=0 :
            continue 
        tokens .append (
        {
        "text":txt ,
        "norm":normalize_text (txt ),
        "line_key":(
        int (data .get ("block_num",[0 ])[i ]or 0 ),
        int (data .get ("par_num",[0 ])[i ]or 0 ),
        int (data .get ("line_num",[0 ])[i ]or 0 ),
        ),
        "y":y ,
        "h":h ,
        }
        )
    if not tokens :
        return 0.0 ,None 

    score =0.0 
    full_text =normalize_text (" ".join (str (t ["text"])for t in tokens ))
    if any (h in full_text for h in TABLE_TITLE_HINTS ):
        score +=5.0 

    nutrient_hits =0 
    for kws in KEYWORDS .values ():
        if any (kw in full_text for kw in kws ):
            nutrient_hits +=1 
    score +=nutrient_hits *2.0 

    numbers =re .findall (r"\b\d{1,4}(?:[.,]\d{1,2})?\b",full_text )
    score +=min (12.0 ,float (len (numbers )))


    by_line :Dict [Tuple [int ,int ,int ],List [Dict [str ,object ]]]={}
    for t in tokens :
        by_line .setdefault (t ["line_key"],[]).append (t )
    title_bottom_y :Optional [int ]=None 
    for line_tokens in by_line .values ():
        line_tokens .sort (key =lambda t :int (t ["y"]))
        line_text =normalize_text (" ".join (str (t ["text"])for t in line_tokens ))
        has_title =any (h in line_text for h in TABLE_TITLE_HINTS )
        if not has_title :
            has_hr =("hranlj"in line_text or "hranj"in line_text or "nutrit"in line_text )
            has_vals =("vred"in line_text or "vrijed"in line_text or "nutrition"in line_text or "facts"in line_text )
            has_title =has_hr and has_vals 
        if has_title :
            bottom =max (int (t ["y"])+int (t ["h"])for t in line_tokens )
            if title_bottom_y is None or bottom <title_bottom_y :
                title_bottom_y =bottom 

    refined_crop =None 
    if title_bottom_y is not None :
        if title_bottom_y <=crop_img .height *0.50 :
            score +=4.0 
        below_tokens =[t for t in tokens if int (t ["y"])>=title_bottom_y -2 ]
        below_text =normalize_text (" ".join (str (t ["text"])for t in below_tokens ))
        below_nums =re .findall (r"\b\d{1,4}(?:[.,]\d{1,2})?\b",below_text )
        score +=min (8.0 ,float (len (below_nums ))*1.2 )
        below_nutrient_hits =0 
        for kws in KEYWORDS .values ():
            if any (kw in below_text for kw in kws ):
                below_nutrient_hits +=1 
        score +=below_nutrient_hits *1.5 


        top =max (0 ,int (title_bottom_y -crop_img .height *0.08 ))
        if crop_img .height -top >=max (60 ,int (crop_img .height *0.40 )):
            refined_crop =crop_img .crop ((0 ,top ,crop_img .width ,crop_img .height ))

    return score ,refined_crop 


def detect_and_crop_with_meta (image_path :Path ,detector_model ,conf :float ):
    try :
        from PIL import Image 
    except Exception :
        return None ,{"detector_used":False ,"reason":"pillow_missing"}
    if detector_model is None :
        return None ,{"detector_used":False ,"reason":"no_detector_model"}

    res =detector_model .predict (str (image_path ),conf =conf ,verbose =False )
    if not res or not hasattr (res [0 ],"boxes")or res [0 ].boxes is None or len (res [0 ].boxes )==0 :
        return None ,{"detector_used":True ,"reason":"no_boxes","detector_candidates":0 }

    img =Image .open (image_path ).convert ("RGB")
    boxes_obj =res [0 ].boxes 
    boxes_xyxy =getattr (boxes_obj ,"xyxy",None )
    confs =getattr (boxes_obj ,"conf",None )


    if boxes_xyxy is not None and len (boxes_xyxy )>0 :
        best_i =0 
        best_conf =-1.0 
        for i in range (len (boxes_xyxy )):
            c =0.0 
            if confs is not None :
                try :
                    c =float (confs [i ].item ()if hasattr (confs [i ],"item")else confs [i ])
                except Exception :
                    c =0.0 
            if c >best_conf :
                best_conf =c 
                best_i =i 
        try :
            x1f ,y1f ,x2f ,y2f =boxes_xyxy [best_i ].tolist ()
            x1 =int (max (0 ,min (img .width -1 ,x1f )))
            y1 =int (max (0 ,min (img .height -1 ,y1f )))
            x2 =int (max (0 ,min (img .width ,x2f )))
            y2 =int (max (0 ,min (img .height ,y2f )))
            x1 ,y1 ,x2 ,y2 =_expand_box_with_padding ((x1 ,y1 ,x2 ,y2 ),img .width ,img .height )
            if x2 >x1 and y2 >y1 :
                area_ratio =((x2 -x1 )*(y2 -y1 ))/max (1.0 ,float (img .width *img .height ))
                crop =img .crop ((x1 ,y1 ,x2 ,y2 ))
                return crop ,{
                "detector_used":True ,
                "detector_candidates":int (len (boxes_xyxy )),
                "detector_box_xyxy":[x1 ,y1 ,x2 ,y2 ],
                "detector_box_area_ratio":round (area_ratio ,6 ),
                "detector_yolo_conf":round (max (0.0 ,float (best_conf )),4 ),
                "detector_selection":"best_confidence_like_YoloVSTruthLabels",
                }
        except Exception :
            pass 


    ranked =_rank_detection_boxes (res [0 ],img .width ,img .height )
    if not ranked :
        return None ,{"detector_used":True ,"reason":"no_ranked_boxes","detector_candidates":0 }

    best_crop =None 
    best_total_score =-1e9 
    best_meta :Dict [str ,object ]={"detector_used":True ,"detector_candidates":len (ranked )}


    for box ,yolo_score in ranked [:4 ]:
        x1 ,y1 ,x2 ,y2 =_expand_box_with_padding (box ,img .width ,img .height )
        if x2 <=x1 or y2 <=y1 :
            continue 
        crop =img .crop ((x1 ,y1 ,x2 ,y2 ))
        ocr_score ,refined_crop =_ocr_score_detection_crop (crop )
        total_score =yolo_score +(ocr_score *1.2 )
        chosen_crop =refined_crop if refined_crop is not None else crop 
        area_ratio =((x2 -x1 )*(y2 -y1 ))/max (1.0 ,float (img .width *img .height ))
        if total_score >best_total_score :
            best_total_score =total_score 
            best_crop =chosen_crop 
            best_meta ={
            "detector_used":True ,
            "detector_candidates":len (ranked ),
            "detector_box_xyxy":[x1 ,y1 ,x2 ,y2 ],
            "detector_box_area_ratio":round (area_ratio ,6 ),
            "detector_yolo_score":round (float (yolo_score ),4 ),
            "detector_ocr_box_score":round (float (ocr_score ),4 ),
            "detector_total_box_score":round (float (total_score ),4 ),
            "detector_refined_by_title":refined_crop is not None ,
            }

    if best_crop is not None :
        return best_crop ,best_meta 


    x1 ,y1 ,x2 ,y2 =_expand_box_with_padding (ranked [0 ][0 ],img .width ,img .height )
    if x2 <=x1 or y2 <=y1 :
        return None ,{"detector_used":True ,"reason":"invalid_best_box_after_padding","detector_candidates":len (ranked )}
    area_ratio =((x2 -x1 )*(y2 -y1 ))/max (1.0 ,float (img .width *img .height ))
    return img .crop ((x1 ,y1 ,x2 ,y2 )),{
    "detector_used":True ,
    "detector_candidates":len (ranked ),
    "detector_box_xyxy":[x1 ,y1 ,x2 ,y2 ],
    "detector_box_area_ratio":round (area_ratio ,6 ),
    "detector_fallback":"yolo_only",
    }


def _preprocess_image_for_ocr_cv2 (image_obj ):

    from PIL import ImageOps ,Image 

    base =image_obj .convert ("RGB")

    base =base .resize (
    (max (1 ,int (base .width *2.0 )),max (1 ,int (base .height *2.0 ))),
    Image .Resampling .LANCZOS ,
    )

    try :
        import cv2 
        import numpy as np 

        arr =np .array (base )
        bgr =cv2 .cvtColor (arr ,cv2 .COLOR_RGB2BGR )
        gray =cv2 .cvtColor (bgr ,cv2 .COLOR_BGR2GRAY )
        gray =cv2 .medianBlur (gray ,3 )
        bw_inv =cv2 .adaptiveThreshold (
        gray ,
        255 ,
        cv2 .ADAPTIVE_THRESH_GAUSSIAN_C ,
        cv2 .THRESH_BINARY_INV ,
        51 ,
        12 ,
        )
        bw_inv =cv2 .morphologyEx (bw_inv ,cv2 .MORPH_OPEN ,np .ones ((2 ,2 ),np .uint8 ))
        bw =cv2 .bitwise_not (bw_inv )
        return Image .fromarray (bw )
    except Exception :
        return ImageOps .grayscale (base )


def _ocr_data_variants (image_obj ,lang :str ="srp+eng"):
    _ =lang 
    return _easyocr_to_data_dict (image_obj )




def _ocr_tokens_from_data (data :Dict [str ,List [object ]]):
    tokens :List [Dict [str ,object ]]=[]
    n =len (data .get ("text",[]))
    for i in range (n ):
        txt =str (data .get ("text",[""])[i ]or "").strip ()
        if not txt :
            continue 
        try :
            conf =float (data .get ("conf",["-1"])[i ])
        except Exception :
            conf =-1.0 
        x =int (data .get ("left",[0 ])[i ]or 0 )
        y =int (data .get ("top",[0 ])[i ]or 0 )
        w =int (data .get ("width",[0 ])[i ]or 0 )
        h =int (data .get ("height",[0 ])[i ]or 0 )
        if w <=0 or h <=0 :
            continue 
        tokens .append (
        {
        "text":txt ,
        "norm":_norm_token (txt ),
        "conf":conf ,
        "x":x ,
        "y":y ,
        "w":w ,
        "h":h ,
        "cx":x +w /2.0 ,
        "cy":y +h /2.0 ,
        "line_key":(
        int (data .get ("block_num",[0 ])[i ]or 0 ),
        int (data .get ("par_num",[0 ])[i ]or 0 ),
        int (data .get ("line_num",[0 ])[i ]or 0 ),
        ),
        }
        )
    return tokens 


def _has_100g_marker (norm_text :str ):
    return (
    ("100"in norm_text and "g"in norm_text )
    or ("100"in norm_text and "ml"in norm_text )
    or norm_text in {"100g","100gr","100ml","100"}
    )


def _find_100g_column_x (tokens :List [Dict [str ,object ]],img_h :int ):
    if not tokens :
        return None 
    by_line :Dict [Tuple [int ,int ,int ],List [Dict [str ,object ]]]={}
    for t in tokens :
        by_line .setdefault (t ["line_key"],[]).append (t )
    candidates :List [Tuple [float ,float ]]=[]
    for line_tokens in by_line .values ():
        line_tokens .sort (key =lambda t :float (t ["x"]))
        line_top =min (float (t ["y"])for t in line_tokens )
        if line_top >img_h *0.45 :
            continue 
        for idx ,t in enumerate (line_tokens ):
            norm =str (t ["norm"])
            if _has_100g_marker (norm ):
                candidates .append ((float (t ["cx"]),float (t ["conf"])))
                continue 
            if norm =="100":
                for j in [idx -1 ,idx +1 ]:
                    if 0 <=j <len (line_tokens ):
                        n2 =str (line_tokens [j ]["norm"])
                        if n2 in {"g","gr","ml"}:
                            candidates .append (
                            ((float (t ["cx"])+float (line_tokens [j ]["cx"]))/2.0 ,float (t ["conf"]))
                            )
                            break 
    if not candidates :
        return None 
    candidates .sort (key =lambda p :p [1 ],reverse =True )
    top =candidates [:5 ]
    xs =sorted (x for x ,_ in top )
    return xs [len (xs )//2 ]


def _token_float_candidates (token_text :str ):
    raw_text =str (token_text or "").strip ()

    if raw_text .startswith ("<"):
        return []
    vals :List [float ]=[]
    for m in re .finditer (r"\d{1,4}(?:[.,]\d{1,2})?",raw_text ):
        try :
            vals .append (parse_float (m .group (0 )))
        except ValueError :
            continue 
    return vals 


def _macro_trailing_g_as_9_candidates (token_text :str ,nutrient :str ):
    if nutrient =="energija":
        return []
    raw =str (token_text or "").strip ().lower ()
    if raw .startswith ("<"):
        return []

    m =re .search (r"(\d{1,3})[.,](\d)(9)\b",raw )
    if not m :
        return []
    try :
        fixed =float (f"{m .group (1 )}.{m .group (2 )}")
    except Exception :
        return []

    if not (0.0 <=fixed <=100.0 ):
        return []
    return [(fixed ,2.4 )]


def _energy_values_from_token_with_bonus (token_text :str ):
    raw =str (token_text or "")
    low =raw .lower ()
    vals =_token_float_candidates (raw )
    out :List [Tuple [float ,float ]]=[]
    if not vals :
        return out 


    if False and "kj"in low and "kcal"in low and len (vals )>=2 :
        for i ,v in enumerate (vals ):
            bonus =0.0 
            if i ==len (vals )-1 :
                bonus +=8.0 
            else :
                bonus -=4.0 
            out .append ((v ,bonus ))
        return out 



    looks_like_kj_fragment =False 
    if len (vals )==1 and "kcal"not in low :
        compact =re .sub (r"\s+","",low )
        if re .search (r"\d(?:[.,]\d+)?(?:kj/|k/|j/)$",compact ):
            looks_like_kj_fragment =True 
        elif "kj"in low :
            looks_like_kj_fragment =True 

    if looks_like_kj_fragment and len (vals )==1 :
        v =vals [0 ]
        out .append ((v ,-2.0 ))
        try :
            kcal =round (v /4.184 ,1 )
        except Exception :
            kcal =None 
        if kcal is not None :
            out .append ((kcal ,6.0 ))
        return out 


    if "kj"in low and "kcal"not in low and len (vals )==1 :
        v =vals [0 ]
        out .append ((v ,-1.5 ))
        try :
            kcal =round (v /4.184 ,1 )
        except Exception :
            kcal =None 
        if kcal is not None :
            out .append ((kcal ,5.0 ))
        return out 


    if "kcal"in low :
        for v in vals :
            out .append ((v ,5.0 ))
        return out 

    for v in vals :
        out .append ((v ,0.0 ))
    return out 


def _energy_neighbor_fragment_bonus (current_token :Dict [str ,object ],row_tokens :List [Dict [str ,object ]]):


    raw =str (current_token .get ("text",""))
    vals =_token_float_candidates (raw )
    if len (vals )!=1 :
        return []

    cx =float (current_token .get ("cx",0.0 ))
    right_neighbors =[t for t in row_tokens if float (t .get ("cx",0.0 ))>cx ]
    if not right_neighbors :
        return []
    right_neighbors .sort (key =lambda t :float (t .get ("cx",0.0 ))-cx )
    nxt =right_neighbors [0 ]
    nxt_text =str (nxt .get ("text","")).lower ()
    compact =re .sub (r"\s+","",nxt_text )
    if not re .match (r"^(kj/|k/|j/|kj|kj\\|)$",compact ):
        return []

    v =vals [0 ]
    out :List [Tuple [float ,float ]]=[(v ,-2.0 )]
    try :
        kcal =round (v /4.184 ,1 )
    except Exception :
        kcal =None 
    if kcal is not None :
        out .append ((kcal ,6.5 ))
    return out 


def _energy_row_kcal_merge_bonus (
current_token :Dict [str ,object ],row_tokens :List [Dict [str ,object ]]
)->List [Tuple [float ,float ]]:
    if not row_tokens :
        return []
    try :
        ordered =sorted (row_tokens ,key =lambda t :float (t .get ("x",0.0 )))
    except Exception :
        ordered =list (row_tokens )

    row_text =" ".join (str (t .get ("text",""))for t in ordered )
    low =row_text .lower ()
    if "kcal"not in low :
        return []


    out :List [Tuple [float ,float ]]=[]
    m_kcal =re .search (r"(\d{1,4}(?:[.,]\d{1,2})?)\s*kcal\b",low )
    if m_kcal :
        try :
            out .append ((parse_float (m_kcal .group (1 )),10.0 ))
        except Exception :
            pass 


    if ("kj"in low or "k/"in low or "j/"in low ):
        vals =_token_float_candidates (row_text )
        if len (vals )>=2 :

            out .append ((vals [0 ],-4.0 ))
            out .append ((vals [1 ],8.0 ))



    best_by_val :Dict [float ,float ]={}
    for v ,b in out :
        if v not in best_by_val or b >best_by_val [v ]:
            best_by_val [v ]=b 
    return [(v ,best_by_val [v ])for v in best_by_val ]


def _line_matches_nutrient (line_tokens :List [Dict [str ,object ]],nutrient :str ):
    line_text =" ".join (str (t ["text"]).lower ()for t in line_tokens )
    if line_tokens :
        min_x =min (float (t ["x"])for t in line_tokens )
        max_x =max (float (t ["x"])+float (t ["w"])for t in line_tokens )
        left_cut =min_x +0.45 *max (1.0 ,(max_x -min_x ))
        left_tokens =[t for t in line_tokens if (float (t ["x"])+float (t ["w"]))<=left_cut ]
        if not left_tokens :
            left_tokens =list (line_tokens )
    else :
        left_tokens =[]
    left_line_text =" ".join (str (t ["text"]).lower ()for t in left_tokens )
    anchor_x :Optional [float ]=None 
    anchor_h :Optional [float ]=None 
    for t in line_tokens :
        n =str (t ["norm"])
        if nutrient =="energija"and (
        "energ"in n or "ener"in n or "energi"in n or "gija"in n or "energy"in n or n in {"kj","kcal"}
        ):
            anchor_x =float (t ["cx"])if anchor_x is None else min (anchor_x ,float (t ["cx"]))
            anchor_h =float (t ["h"])if anchor_h is None else max (anchor_h ,float (t ["h"]))
        if nutrient =="masti"and ("masti"in n or "mast"in n or "fat"in n ):
            anchor_x =float (t ["cx"])if anchor_x is None else min (anchor_x ,float (t ["cx"]))
            anchor_h =float (t ["h"])if anchor_h is None else max (anchor_h ,float (t ["h"]))
        if nutrient =="ugljeni_hidrati"and (
        "ugljen"in n or "uglje"in n or "uglj"in n or "hidrati"in n or "hidrat"in n or "hidr"in n or "drati"in n or "carb"in n or "sachar"in n 
        ):
            anchor_x =float (t ["cx"])if anchor_x is None else min (anchor_x ,float (t ["cx"]))
            anchor_h =float (t ["h"])if anchor_h is None else max (anchor_h ,float (t ["h"]))
        if nutrient =="proteini"and (
        "protein"in n or "protei"in n or "prote"in n or "prot"in n or "tein"in n or "belanc"in n or "bjelanc"in n 
        ):
            anchor_x =float (t ["cx"])if anchor_x is None else min (anchor_x ,float (t ["cx"]))
            anchor_h =float (t ["h"])if anchor_h is None else max (anchor_h ,float (t ["h"]))
    if anchor_x is not None :


        return True ,anchor_x 

    extra_frags ={
    "energija":["ener","energ","energi","gija","energy"],
    "masti":["masti","mast","fat"],
    "ugljeni_hidrati":["ugl","uglj","ugljen","uglje","hidr","hidrat","hidrati","drati","carb","sachar"],
    "proteini":["prot","tein","prote","protei","protein","belan","belanc","bjelan","bjelanc"],
    }

    if any (kw in line_text for kw in KEYWORDS [nutrient ])or any (f in left_line_text for f in extra_frags [nutrient ]):
        return True ,min (float (t ["cx"])for t in left_tokens or line_tokens )
    return False ,None 


def _is_sub_nutrient_detail_line (line_tokens :List [Dict [str ,object ]],nutrient :str ):


    raw =" ".join (str (t ["text"]).lower ()for t in line_tokens )
    norm =" ".join (str (t .get ("norm",""))for t in line_tokens )

    has_of_which =(
    "od kojih"in raw 
    or "odkojih"in norm 
    or "od tega"in raw 
    or "odtega"in norm 
    or "od tego"in raw 
    or "odtego"in norm 
    or "of which"in raw 
    or "z toho"in raw 
    or "iz kojih"in raw 
    )

    sugar_markers =[
    "secer","šećer","seceri","šećeri","sugar","sugars","zahar","cukr","cukor"
    ]
    saturated_markers =[
    "zasic","zasić","satur","saturated","masne kiseline","fatty acids"
    ]

    sugar_markers .extend (["sladkor","sladkorji","sladkori"])
    saturated_markers .extend (["nasic","nasicene","saturati","acizi"])

    if nutrient =="ugljeni_hidrati":
        if has_of_which and any (m in raw for m in sugar_markers ):
            return True 
        if any (m .replace (" ","")in norm for m in ["secer","seceri","sugar","sugars","zahar","cukr","cukor"]):
            return True 

    if nutrient =="masti":
        if has_of_which and any (m in raw for m in saturated_markers ):
            return True 
        if any (m .replace (" ","")in norm for m in ["zasic","satur","fattyacids"]):
            return True 

    return False 


def _is_footer_or_reference_line (line_tokens :List [Dict [str ,object ]],img_h :float ):
    if not line_tokens :
        return False 
    line_top =min (float (t ["y"])for t in line_tokens )
    line_bottom =max (float (t ["y"])+float (t ["h"])for t in line_tokens )
    raw =" ".join (str (t ["text"]).lower ()for t in line_tokens )
    norm =" ".join (str (t .get ("norm",""))for t in line_tokens )

    footer_markers =[
    "referenc","reference","ri","unos","prosecn","prosje","adult","odras","daily","intake"
    ]
    if any (m in raw for m in footer_markers )or any (m in norm for m in footer_markers ):
        return True 

    if line_top >img_h *0.86 :

        if any (k in raw for k in ["energ","mast","fat","uglj","hidrat","protein","belan","bjelan","so","sol","salt"]):
            return False 
        return True 
    if line_bottom >img_h *0.9 :
        return True 
    return False 


def _extract_nutrients_from_ocr_layout (data :Dict [str ,List [object ]],img_w :int ,img_h :int ):
    tokens =_ocr_tokens_from_data (data )
    out :Dict [str ,Optional [float ]]={k :None for k in NUTRIENTS }
    if not tokens :
        return out 

    by_line :Dict [Tuple [int ,int ,int ],List [Dict [str ,object ]]]={}
    for t in tokens :
        by_line .setdefault (t ["line_key"],[]).append (t )
    line_items :List [Tuple [Tuple [int ,int ,int ],List [Dict [str ,object ]]]]=[]
    for k ,line_tokens in by_line .items ():
        line_tokens .sort (key =lambda t :float (t ["x"]))
        line_items .append ((k ,line_tokens ))

    header_x =_find_100g_column_x (tokens ,img_h )
    x_tol =max (18.0 ,img_w *0.20 )
    selected_row_cy :Dict [str ,float ]={}
    prev_nutrient ={
    "masti":"energija",
    "ugljeni_hidrati":"masti",
    "proteini":"ugljeni_hidrati",
    }

    for nutrient in NUTRIENTS :
        best_score =-10_000.0 
        best_val :Optional [float ]=None 
        best_row_for_nutrient :Optional [float ]=None 
        for _ ,line_tokens in line_items :
            if _is_footer_or_reference_line (line_tokens ,float (img_h )):
                continue 
            matches ,anchor_x =_line_matches_nutrient (line_tokens ,nutrient )
            if not matches :
                continue 
            if _is_sub_nutrient_detail_line (line_tokens ,nutrient ):
                continue 
            if anchor_x is None :
                anchor_x =min (float (t ["cx"])for t in line_tokens )

            anchor_h_vals :List [float ]=[]
            for t in line_tokens :
                n =str (t ["norm"])
                if nutrient =="energija"and ("energ"in n or "ener"in n or "energi"in n or "gija"in n or "energy"in n or n in {"kj","kcal"}):
                    anchor_h_vals .append (float (t ["h"]))
                elif nutrient =="masti"and ("masti"in n or "mast"in n or "fat"in n ):
                    anchor_h_vals .append (float (t ["h"]))
                elif nutrient =="ugljeni_hidrati"and ("ugljen"in n or "uglje"in n or "uglj"in n or "hidrati"in n or "hidrat"in n or "hidr"in n or "drati"in n or "carb"in n or "sachar"in n ):
                    anchor_h_vals .append (float (t ["h"]))
                elif nutrient =="proteini"and ("protein"in n or "protei"in n or "prote"in n or "prot"in n or "tein"in n or "belanc"in n or "bjelanc"in n ):
                    anchor_h_vals .append (float (t ["h"]))
            row_cy =sum (float (t ["cy"])for t in line_tokens )/max (1 ,len (line_tokens ))
            row_h =max (float (t ["h"])for t in line_tokens )
            current_line_key =line_tokens [0 ]["line_key"]if line_tokens else None 
            anchor_h =max (anchor_h_vals )if anchor_h_vals else row_h 
            row_order_score =0.0 
            prev_name =prev_nutrient .get (nutrient )
            if prev_name and prev_name in selected_row_cy :
                prev_cy =float (selected_row_cy [prev_name ])

                if row_cy >prev_cy +max (3.0 ,row_h *0.15 ):
                    row_order_score +=3.0 
                elif row_cy <prev_cy -max (3.0 ,row_h *0.10 ):
                    row_order_score -=10.0 
                else :
                    row_order_score -=2.0 

            if nutrient =="energija":
                y_tol =max (10.0 ,anchor_h *1.0 )
            elif nutrient =="proteini":
                y_tol =max (6.0 ,row_h *0.75 )
            else :
                y_tol =max (7.0 ,anchor_h *0.9 )


            nearby_tokens =[t for t in tokens if abs (float (t ["cy"])-row_cy )<=y_tol ]
            if not nearby_tokens :
                nearby_tokens =line_tokens 

            if header_x is not None :

                col_tokens =[t for t in nearby_tokens if abs (float (t ["cx"])-header_x )<=x_tol ]
                use_100g_band =len (col_tokens )>0 
                candidate_tokens =col_tokens if use_100g_band else nearby_tokens 
            else :
                use_100g_band =False 
                candidate_tokens =nearby_tokens 


            right_tokens =[t for t in candidate_tokens if float (t ["cx"])>anchor_x ]
            if right_tokens :
                if use_100g_band :

                    candidate_tokens =right_tokens 
                else :

                    right_number_tokens =[]
                    for t in right_tokens :
                        raw_text =str (t ["text"])
                        if "%"in raw_text :
                            continue 
                        if _token_float_candidates (raw_text ):
                            right_number_tokens .append (t )
                    if right_number_tokens :
                        right_number_tokens .sort (key =lambda t :float (t ["cx"])-anchor_x )
                        candidate_tokens =[right_number_tokens [0 ]]
                    else :
                        candidate_tokens =[]


            if nutrient =="proteini"and current_line_key is not None and candidate_tokens :
                same_line_number_tokens =[]
                for t2 in candidate_tokens :
                    if t2 .get ("line_key")!=current_line_key :
                        continue 
                    rtxt2 =str (t2 .get ("text",""))
                    if "%"in rtxt2 :
                        continue 
                    if not _token_float_candidates (rtxt2 ):
                        continue 
                    same_line_number_tokens .append (t2 )
                if same_line_number_tokens :
                    candidate_tokens =same_line_number_tokens 

            for t in candidate_tokens :
                raw_text =str (t ["text"])
                if "%"in raw_text :
                    continue 

                if header_x is not None :
                    tx0 =float (t ["cx"])
                    max_right =header_x +(x_tol *(2.3 if nutrient =="energija"else 1.6 ))
                    if tx0 >max_right :
                        continue 
                if nutrient =="energija":
                    val_bonus_pairs =_energy_values_from_token_with_bonus (raw_text )
                    nb_pairs =_energy_neighbor_fragment_bonus (t ,nearby_tokens )
                    if nb_pairs :
                        val_bonus_pairs .extend (nb_pairs )
                    row_merge_pairs =_energy_row_kcal_merge_bonus (t ,nearby_tokens )
                    if row_merge_pairs :
                        val_bonus_pairs .extend (row_merge_pairs )
                else :
                    val_bonus_pairs =[(v ,0.0 )for v in _token_float_candidates (raw_text )]
                    val_bonus_pairs .extend (_macro_trailing_g_as_9_candidates (raw_text ,nutrient ))

                    if "."not in raw_text and ","not in raw_text :
                        extra_macro_pairs =[]
                        for v ,_ in val_bonus_pairs :
                            if 10 <=v <=999 :
                                bonus =0.0 
                                if nutrient =="proteini":
                                    bonus =2.2 
                                elif nutrient =="masti":
                                    bonus =1.4 
                                elif nutrient =="ugljeni_hidrati":
                                    bonus =0.8 
                                extra_macro_pairs .append ((v /10.0 ,bonus ))
                        val_bonus_pairs .extend (extra_macro_pairs )

                    elif nutrient =="masti":
                        extra_macro_pairs =[]
                        for v ,_ in val_bonus_pairs :
                            if 3.0 <=v <=9.9 :
                                extra_macro_pairs .append ((v /10.0 ,1.6 ))
                        val_bonus_pairs .extend (extra_macro_pairs )
                if not val_bonus_pairs :
                    continue 
                for raw_val ,extra_energy_bonus in val_bonus_pairs :
                    val =_sanitize_prediction (nutrient ,raw_val )
                    if val is None :
                        continue 
                    score =0.0 
                    score +=row_order_score 
                    tx =float (t ["cx"])
                    ty =float (t ["cy"])
                    if tx >anchor_x :
                        score +=3.0 
                    else :
                        score -=6.0 
                        if not use_100g_band :
                            continue 
                    if header_x is not None and use_100g_band :
                        dx =abs (tx -header_x )

                        score +=max (-7.0 ,9.0 -(dx /max (12.0 ,img_w *0.04 )))
                        if dx <=x_tol :
                            score +=4.5 
                        else :
                            score -=5.0 
                    elif not use_100g_band :

                        dx_right =tx -anchor_x 
                        score +=max (-8.0 ,8.0 -(dx_right /max (10.0 ,img_w *0.03 )))


                        if header_x is not None :
                            dxh =abs (tx -header_x )
                            score +=max (-4.0 ,3.0 -(dxh /max (14.0 ,img_w *0.08 )))
                    if current_line_key is not None :
                        if t .get ("line_key")==current_line_key :
                            score +=3.0 
                        else :
                            score -=6.0 

                    dy =abs (ty -row_cy )
                    score +=max (-2.0 ,2.5 -(dy /max (8.0 ,row_h *0.5 )))



                    row_neighbor_tokens =[rt for rt in nearby_tokens if abs (float (rt ["cy"])-ty )<=max (6.0 ,row_h *0.35 )]
                    has_percent_neighbor =False 
                    for rt in row_neighbor_tokens :
                        if rt is t :
                            continue 
                        rtxt =str (rt ["text"]).strip ()
                        if rtxt =="%"and abs (float (rt ["cx"])-tx )<=max (28.0 ,img_w *0.05 ):
                            has_percent_neighbor =True 
                            break 
                    if "%"in raw_text or has_percent_neighbor :
                        score -=8.0 
                    if tx >=img_w *0.86 :
                        score -=6.0 


                    if header_x is not None and tx >(header_x +x_tol *1.25 ):
                        score -=5.5 


                    if nutrient !="energija"and (","in raw_text or "."in raw_text ):
                        score +=1.5 
                    if nutrient in {"masti","proteini"}and (","in raw_text or "."in raw_text ):
                        score +=1.0 
                    if nutrient =="proteini"and ("."not in raw_text and ","not in raw_text )and val >=20 :
                        score -=2.5 
                    if nutrient =="masti"and ("."not in raw_text and ","not in raw_text )and val >=35 :
                        score -=1.8 
                    if nutrient =="energija"and ("kcal"in raw_text .lower ()or "kj"in raw_text .lower ()):
                        score +=1.0 
                    if nutrient =="energija":
                        score +=float (extra_energy_bonus )
                        if raw_val >800 and "kcal"not in raw_text .lower ():

                            if abs (val -round (raw_val /4.184 ,1 ))<1.0 :
                                score +=5.0 
                            elif val >800 :
                                score -=10.0 
                        if val >750 and "kcal"not in raw_text .lower ():
                            score -=4.0 
                        if val >=850 and "kcal"not in raw_text .lower ():
                            score -=6.0 
                        if val <80 and "kcal"not in raw_text .lower ():
                            score -=1.5 

                    if 95 <=raw_val <=105 and nutrient !="energija":
                        score -=3.0 

                    if nutrient =="energija"and 40 <=val <=850 :
                        score +=1.5 
                    if nutrient !="energija"and 0 <=val <=80 :
                        score +=1.0 
                    if nutrient in {"masti","proteini"}and 0 <=val <=40 :
                        score +=0.8 
                    if score >best_score :
                        best_score =score 
                        best_val =val 
                        best_row_for_nutrient =row_cy 
        out [nutrient ]=best_val 
        if best_val is not None and best_row_for_nutrient is not None :
            selected_row_cy [nutrient ]=float (best_row_for_nutrient )
    return out 


def extract_nutrients_from_image (image_obj ,lang :str ="srp+eng"):
    data ,best_text =_ocr_data_variants (image_obj ,lang =lang )
    layout_pred =_extract_nutrients_from_ocr_layout (data ,image_obj .width ,image_obj .height )
    text_pred =extract_nutrients_from_text (best_text )

    out :Dict [str ,Optional [float ]]={}
    for k in NUTRIENTS :
        out [k ]=layout_pred .get (k )if layout_pred .get (k )is not None else text_pred .get (k )
    return out 


def _prediction_quality_score (pred :Dict [str ,Optional [float ]]):
    score =0.0 
    present =0 
    for k in NUTRIENTS :
        v =pred .get (k )
        if v is None :
            continue 
        present +=1 
        score +=10.0 
        fv =float (v )
        if k =="energija":
            if 20 <=fv <=950 :
                score +=3.0 
            if 40 <=fv <=850 :
                score +=2.0 
        else :
            if 0 <=fv <=100 :
                score +=2.0 
            if 0 <=fv <=80 :
                score +=1.0 
    if present >=3 :
        score +=4.0 
    if present ==4 :
        score +=3.0 
    return score 


def _norm_token (token :str ):
    return re .sub (r"[^a-z0-9]+","",token .lower ())


def normalize_text (s :str ):
    return " ".join (s .lower ().replace ("\n"," ").split ())


def _sanitize_prediction (nutrient :str ,value :float ):

    if nutrient =="energija":
        if value <=0 :
            return None 

        if value >900 :
            kcal =value /4.184 
            if 20 <=kcal <=950 :
                return round (kcal ,1 )

        v =value 
        while v >950 :
            v /=10.0 
        return round (v ,1 )if 20 <=v <=950 else None 



    v =value 
    if v <=0 :
        return None 
    if nutrient =="proteini"and v >35 :
        while v >35 :
            v /=10.0 
        return round (v ,1 )if 0 <=v <=50 else None 
    if nutrient =="masti"and v >50 :
        while v >50 :
            v /=10.0 
        return round (v ,1 )if 0 <=v <=50 else None 
    if nutrient =="ugljeni_hidrati"and v >100 :
        while v >100 :
            v /=10.0 
        return round (v ,1 )if 0 <=v <=100 else None 
    while v >100 :
        v /=10.0 
    return round (v ,1 )if 0 <=v <=100 else None 


def _extract_candidates (text_norm :str ,keywords :List [str ]):

    candidates :List [Tuple [float ,int ]]=[]
    fragments ={
    "energija":["energ","energi","energy"],
    "masti":["masti","mast","fat"],
    "ugljeni_hidrati":["ugljen","uglje","hidrat","hidrati","carb","sachar"],
    "proteini":["protein","protei","prote","prot","belanc","bjelanc"],
    }

    active_frags :List [str ]=[]
    for nutrient_name ,frag_list in fragments .items ():
        if keywords is KEYWORDS .get (nutrient_name ):
            active_frags =frag_list 
            break 

    all_keys =list (dict .fromkeys (list (keywords )+active_frags ))
    for kw in all_keys :

        patterns =[
        (rf"{re .escape (kw )}[^0-9]{{0,24}}([0-9]{{1,4}}(?:[.,][0-9]{{1,2}})?)\s*(kcal|kj|g)?",3 ),
        (rf"{re .escape (kw )}[^0-9]{{0,40}}([0-9]{{1,4}}(?:[.,][0-9]{{1,2}})?)",2 ),
        ]
        for pat ,base_score in patterns :
            for m in re .finditer (pat ,text_norm ,re .IGNORECASE ):
                raw =m .group (1 )
                unit =(m .group (2 )or "").lower ()if len (m .groups ())>1 else ""
                try :
                    val =parse_float (raw )
                except ValueError :
                    continue 
                score =base_score 
                if unit in {"kcal","kj","g"}:
                    score +=2 
                candidates .append ((val ,score ))
    return candidates 


def extract_nutrients_from_text (text :str ):
    t =normalize_text (text )
    out :Dict [str ,Optional [float ]]={}
    for nutrient in NUTRIENTS :
        candidates =_extract_candidates (t ,KEYWORDS [nutrient ])
        best :Optional [float ]=None 
        best_score =-1 
        for raw_val ,score in candidates :
            val =_sanitize_prediction (nutrient ,raw_val )
            if val is None :
                continue 

            if nutrient =="energija"and 40 <=val <=850 :
                score +=2 
            if nutrient !="energija"and 0 <=val <=80 :
                score +=2 
            if score >best_score :
                best_score =score 
                best =val 
        out [nutrient ]=best 
    return out 


def calculate_intake (values_per_100g :Dict [str ,Optional [float ]],grams :float ):
    factor =grams /100.0 
    result :Dict [str ,Optional [float ]]={}
    for k in NUTRIENTS :
        v =values_per_100g .get (k )
        result [k ]=None if v is None else round (v *factor ,3 )
    return result 


def _refine_macros_with_energy_consistency (pred :Dict [str ,Optional [float ]]):


    try :
        e =pred .get ("energija")
        m =pred .get ("masti")
        uh =pred .get ("ugljeni_hidrati")
        p =pred .get ("proteini")
        if e is None or m is None or uh is None or p is None :
            return pred 
        e_val =float (e )
        m_val =float (m )
        uh_val =float (uh )
        p_val =float (p )
    except Exception :
        return pred 

    if not (120 <=e_val <=950 ):
        return pred 

    def kcal_est (mv ,uhv ,pv ):
        return 9.0 *mv +4.0 *uhv +4.0 *pv 

    out =dict (pred )
    base_diff =abs (kcal_est (m_val ,uh_val ,p_val )-e_val )


    if m_val >=1.0 :
        m10 =round (m_val /10.0 ,1 )
        diff_m10 =abs (kcal_est (m10 ,uh_val ,p_val )-e_val )
        if diff_m10 +20 <base_diff :
            out ["masti"]=m10 
            m_val =m10 
            base_diff =diff_m10 


    if p_val >=1.0 :
        p10 =round (p_val /10.0 ,1 )
        diff_p10 =abs (kcal_est (m_val ,uh_val ,p10 )-e_val )
        if diff_p10 +20 <base_diff :
            out ["proteini"]=p10 

    return out 


def evaluate_predictions (rows :List [Dict [str ,object ]]):
    mae ={k :[]for k in NUTRIENTS }
    signed_err ={k :[]for k in NUTRIENTS }
    found ={k :0 for k in NUTRIENTS }
    total =0 
    within_20 ={k :0 for k in NUTRIENTS }
    for row in rows :
        gt =row .get ("truth")
        pred =row .get ("pred")
        if not gt or not pred :
            continue 
        total +=1 
        for k in NUTRIENTS :
            pv =pred .get (k )
            gv =gt .get (k )
            if pv is not None and gv is not None :
                found [k ]+=1 
                diff =float (pv )-float (gv )
                err =abs (diff )
                signed_err [k ].append (diff )
                mae [k ].append (err )
                if float (gv )!=0.0 :
                    rel =err /abs (float (gv ))
                    if rel <=0.20 :
                        within_20 [k ]+=1 
    return {
    "samples_with_truth":total ,
    "found_count":{k :found [k ]for k in NUTRIENTS },
    "coverage":{k :(found [k ]/total if total else 0.0 )for k in NUTRIENTS },
    "mae":{k :(statistics .mean (mae [k ])if mae [k ]else None )for k in NUTRIENTS },
    "sum_error_signed":{k :(sum (signed_err [k ])if signed_err [k ]else None )for k in NUTRIENTS },
    "mean_error_signed":{k :(statistics .mean (signed_err [k ])if signed_err [k ]else None )for k in NUTRIENTS },
    "avg_diff_from_sum_div_count":{
    k :((sum (signed_err [k ])/found [k ])if found [k ]and signed_err [k ]else None )for k in NUTRIENTS 
    },
    "accuracy_within_20_percent":{
    k :(within_20 [k ]/found [k ]if found [k ]else 0.0 )for k in NUTRIENTS 
    },
    }


def _round_floats_for_print (obj ,ndigits :int =2 ):
    if isinstance (obj ,float ):
        return round (obj ,ndigits )
    if isinstance (obj ,dict ):
        return {k :_round_floats_for_print (v ,ndigits )for k ,v in obj .items ()}
    if isinstance (obj ,list ):
        return [_round_floats_for_print (v ,ndigits )for v in obj ]
    return obj 


def find_correct_image_ids (rows ,rel_tol =0.10 ):
    ok_ids =[]
    for row in rows :
        image_id =row .get ("image_id")
        pred =row .get ("pred")or {}
        gt =row .get ("truth")or {}
        if not image_id or not gt :
            continue 

        all_ok =True 
        for k in NUTRIENTS :
            pv =pred .get (k )
            gv =gt .get (k )
            if pv is None or gv is None :
                all_ok =False 
                break 
            gvf =float (gv )
            pvf =float (pv )
            if gvf ==0 :
                if abs (pvf -gvf )>1e-6 :
                    all_ok =False 
                    break 
            else :
                if abs (pvf -gvf )/abs (gvf )>rel_tol :
                    all_ok =False 
                    break 
        if all_ok :
            ok_ids .append (str (image_id ))
    return ok_ids 


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
            print (json .dumps (_round_floats_for_print (eval_all [key ],2 ),ensure_ascii =False ,indent =2 ))
    ocr_source_counts :Dict [str ,int ]={}
    for row in results :
        src =str (row .get ("ocr_source")or "unknown")
        ocr_source_counts [src ]=ocr_source_counts .get (src ,0 )+1 
    print ("ocr_source_counts:")
    print (json .dumps (_round_floats_for_print (ocr_source_counts ,2 ),ensure_ascii =False ,indent =2 ))
    print ("")
    print (f"Broj slika tacno uradjenih (sva 4 nutrienta unutar +-20%): {len (correct_ids_20 )}")
    print (f"Broj slika tacno uradjenih (3/4 nutrienta unutar +-20%): {len (correct_ids_3of4_20 )}")


def main ():
    print ("Info: trenutno_easyocr.py koristi EasyOCR (isti parser/metrike kao trenutno.py).")
    _get_easyocr_reader ()
    print ("Info: EasyOCR inicijalizovan.")
    # Ako hoces od 0 (YOLO trening + OCR): stavi mode = "train" i detector_weights_path = "".
    # Ako hoces sa postojecim modelom: stavi detector_weights_path na .\\outputs\\runs\\nutrition_yolov8n\\weights\\best.pt.
    project_dir =Path (".").resolve ()
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
    detector_weights =(Path (detector_weights_path ).resolve ()if detector_weights_path .strip ()else None ),
    )
    run_full_pipeline (cfg )


if __name__ =="__main__":
    main ()
