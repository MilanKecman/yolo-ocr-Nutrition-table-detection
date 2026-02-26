import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .ocr_easy import _easyocr_to_data_dict
    from .parser_heuristics import KEYWORDS, normalize_text
except ImportError:
    from ocr_easy import _easyocr_to_data_dict
    from parser_heuristics import KEYWORDS, normalize_text


TABLE_TITLE_HINTS =[
"hranljive vrednosti",
"hranjive vrijednosti",
"nutritivne vrednosti",
"nutritivne vrijednosti",
"nutrition facts",
"nutritional values",
"nutrition",
]

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

