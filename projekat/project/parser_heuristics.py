import re
from typing import Dict, List, Optional, Tuple

try:
    from .data_io import parse_float
    from .ocr_easy import _easyocr_to_data_dict
except ImportError:
    from data_io import parse_float
    from ocr_easy import _easyocr_to_data_dict


NUTRIENTS =["energija","masti","ugljeni_hidrati","proteini"]

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

