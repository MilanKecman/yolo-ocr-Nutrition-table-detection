import json 
import re 
from pathlib import Path 

from PIL import Image ,ImageDraw 


def broj_slike (ime ):
    return int (re .sub (r"\D","",ime )or "0")


def procitaj_labelu_yolo (putanja ):
    if not putanja .exists ():
        return None 
    txt =putanja .read_text (encoding ="utf-8",errors ="ignore").strip ()
    if txt =="":
        return None 
    delovi =txt .splitlines ()[0 ].split ()
    if len (delovi )!=5 :
        return None 
    try :
        klasa =int (delovi [0 ])
        cx =float (delovi [1 ])
        cy =float (delovi [2 ])
        w =float (delovi [3 ])
        h =float (delovi [4 ])
    except Exception :
        return None 
    return klasa ,cx ,cy ,w ,h 


def yolo_u_piksele (sirina ,visina ,cx ,cy ,w ,h ):
    bw =w *sirina 
    bh =h *visina 
    x1 =int (cx *sirina -bw /2 )
    y1 =int (cy *visina -bh /2 )
    x2 =int (cx *sirina +bw /2 )
    y2 =int (cy *visina +bh /2 )
    x1 =max (0 ,min (sirina -1 ,x1 ))
    y1 =max (0 ,min (visina -1 ,y1 ))
    x2 =max (0 ,min (sirina ,x2 ))
    y2 =max (0 ,min (visina ,y2 ))
    return x1 ,y1 ,x2 ,y2 


def iou (box1 ,box2 ):
    x1a ,y1a ,x2a ,y2a =box1 
    x1b ,y1b ,x2b ,y2b =box2 
    ix1 =max (x1a ,x1b )
    iy1 =max (y1a ,y1b )
    ix2 =min (x2a ,x2b )
    iy2 =min (y2a ,y2b )
    iw =max (0 ,ix2 -ix1 )
    ih =max (0 ,iy2 -iy1 )
    presek =iw *ih 
    p1 =max (0 ,x2a -x1a )*max (0 ,y2a -y1a )
    p2 =max (0 ,x2b -x1b )*max (0 ,y2b -y1b )
    unija =p1 +p2 -presek 
    if unija <=0 :
        return 0 
    return presek /unija 


def main ():

    project_dir =Path (__file__ ).resolve ().parent .parent 
    scripts_dir =Path (__file__ ).resolve ().parent 
    images_dir =project_dir /"cleanData2"
    labels_dir =project_dir /"labels"
    model_path =project_dir /"outputs"/"runs"/"nutrition_yolov8n"/"weights"/"best.pt"
    out_dir =scripts_dir /"debug_yolo_pred_vs_labels"
    limit =200 
    conf =0.25 
    split_ids_path =project_dir /"outputs"/"detector_dataset"/"split_ids.json"

    try :
        from ultralytics import YOLO 
    except Exception :
        print ("Nedostaje ultralytics. Instaliraj: pip install ultralytics")
        return 

    if not model_path .exists ():
        print ("Model ne postoji:",model_path )
        print ("Prvo pokreni: python .\\run_full_project.py --mode train")
        return 

    if not images_dir .exists ():
        print ("Folder sa slikama ne postoji:",images_dir )
        return 

    out_dir .mkdir (parents =True ,exist_ok =True )

    model =YOLO (str (model_path ))
    slike =sorted (images_dir .glob ("slika*.jpeg"),key =lambda p :broj_slike (p .stem ))
    if len (slike )==0 :
        print ("Nema slika u folderu:",images_dir )
        return 

    if split_ids_path .exists ():
        try :
            split_ids =json .loads (split_ids_path .read_text (encoding ="utf-8"))
            test_ids =set (str (x ).lower ()for x in split_ids .get ("test",[]))
            if test_ids :
                slike =[p for p in slike if p .stem .lower ()in test_ids ]
                limit =min (limit ,len (slike ))
                print ("Koristim samo test split iz:",split_ids_path )
                print ("Broj test slika:",len (slike ))
        except Exception as e :
            print ("Upozorenje: nije uspelo ucitavanje split_ids.json:",e )

    broj_gt =0 
    broj_pred =0 
    broj_oba =0 
    iou_vrednosti =[]

    for slika_putanja in slike [:limit ]:
        img =Image .open (slika_putanja ).convert ("RGBA")
        sloj =Image .new ("RGBA",img .size ,(0 ,0 ,0 ,0 ))
        crtaj =ImageDraw .Draw (sloj )


        gt_box =None 
        label_putanja =labels_dir /(slika_putanja .stem .lower ()+".txt")
        parsed =procitaj_labelu_yolo (label_putanja )
        if parsed is not None :
            broj_gt +=1 
            _ ,cx ,cy ,w ,h =parsed 
            gt_box =yolo_u_piksele (img .width ,img .height ,cx ,cy ,w ,h )


        pred_box =None 
        pred_conf =None 
        rezultat =model .predict (str (slika_putanja ),conf =conf ,verbose =False )
        if rezultat and hasattr (rezultat [0 ],"boxes")and rezultat [0 ].boxes is not None and len (rezultat [0 ].boxes )>0 :
            boxes =rezultat [0 ].boxes .xyxy 
            confs =getattr (rezultat [0 ].boxes ,"conf",None )
            najbolji_i =0 
            najbolji_conf =-1 
            for i in range (len (boxes )):
                c =0 
                if confs is not None :
                    try :
                        c =float (confs [i ].item ()if hasattr (confs [i ],"item")else confs [i ])
                    except Exception :
                        c =0 
                if c >najbolji_conf :
                    najbolji_conf =c 
                    najbolji_i =i 
            x1f ,y1f ,x2f ,y2f =boxes [najbolji_i ].tolist ()
            x1 =int (max (0 ,min (img .width -1 ,x1f )))
            y1 =int (max (0 ,min (img .height -1 ,y1f )))
            x2 =int (max (0 ,min (img .width ,x2f )))
            y2 =int (max (0 ,min (img .height ,y2f )))
            pred_box =(x1 ,y1 ,x2 ,y2 )
            pred_conf =max (0 ,float (najbolji_conf ))
            broj_pred +=1 

        if pred_box is not None :
            x1 ,y1 ,x2 ,y2 =pred_box 
            crtaj .rectangle ([x1 ,y1 ,x2 ,y2 ],fill =(255 ,255 ,0 ,35 ),outline =(255 ,255 ,0 ,220 ),width =3 )

        if gt_box is not None :
            x1 ,y1 ,x2 ,y2 =gt_box 
            crtaj .rectangle ([x1 ,y1 ,x2 ,y2 ],outline =(0 ,255 ,0 ,220 ),width =3 )

        if gt_box is not None and pred_box is not None :
            broj_oba +=1 
            i =iou (gt_box ,pred_box )
            iou_vrednosti .append (i )
            txt ="IoU="+str (round (i ,2 ))+" conf="+str (round (pred_conf ,2 ))
            crtaj .rectangle ([4 ,4 ,230 ,30 ],fill =(0 ,0 ,0 ,160 ))
            crtaj .text ((8 ,8 ),txt ,fill =(255 ,255 ,255 ,230 ))

        izlaz =Image .alpha_composite (img ,sloj ).convert ("RGB")
        izlaz .save (out_dir /slika_putanja .name ,quality =95 )

    avg_iou =None 
    if len (iou_vrednosti )>0 :
        avg_iou =sum (iou_vrednosti )/len (iou_vrednosti )

    print ("Model:",model_path )
    print ("Slike:",images_dir )
    print ("Labele:",labels_dir )
    print ("Izlaz:",out_dir )
    print ("Broj sacuvanih slika:",min (len (slike ),limit ))
    print ("GT labela:",broj_gt )
    print ("YOLO predikcija:",broj_pred )
    print ("Oba postoje:",broj_oba )
    print ("Prosecan IoU:",avg_iou )


if __name__ =="__main__":
    main ()
