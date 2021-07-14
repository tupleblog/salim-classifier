# Salim Classifier

**วัตถุประสงค์โครงงาน:** ทุกวันนี้หาเพื่อนที่รักชาติ ศาสนา พระมหากษัตริย์ รัฐบาลยากเหลือเกิน มีแต่พวกสามกีบ ควายแดงคอยจ้องจะทำร้าย
ทางทีมของเราจึงสร้าง AI มาเพื่อช่วยหาเพื่อนสลิ่มที่หลงเหลืออยู่น้อยยิ่งนักในสังคมไทย เพื่อเป็นแนวทางในการสร้างสังคมสลิ่มที่แข็งแรงต่อไป

## วิธีการใช้งาน

สามารถลง `transfomers` จาก Huggingface เพื่อใช้งานได้ต่อไปนี้

``` py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# download model
tokenizer = AutoTokenizer.from_pretrained("tupleblog/salim-classifier")
model = AutoModelForSequenceClassification.from_pretrained("tupleblog/salim-classifier")

# using pipeline
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
text = "จิตไม่ปกติ วันๆคอยแต่ให้คนเสี้ยมทะเลาะกันด่ากัน คอยจ้องแต่จะเล่นงานรัฐบาล ความคดด้านลบ"
classifier(text)
```

## การเก็บข้อมูล

Annotate ข้อมูลจากเพจ Facebook ที่มีชาวสลิ่มเข้าไปคอมเม้น คัดเลือกชุดประโยคที่มีความยาวมากกว่า 50 ตัวอักษร
และนำมาเทรนโมเดลด้วย WangchanBERTa ข้อมูลอาจมีความ bias เนื่องจากทางทีมงานเป็นผู้เก็บข้อมูล อาจจะไม่สามารถเก็บข้อมูล

## ทดลองใช้งานผ่าน HuggingFace

ท่านสามารถทดลองใช้งานผ่าน HuggingFace โดยใส่คอมเม้นจาก Facebook เข้าไปในช่องได้จาก
[huggingface.co/tupleblog/salim-classifier](https://huggingface.co/tupleblog/salim-classifier)

ตัวอย่างประโยค
- รัฐรับผิดชอบทุกชีวิตไม่ได้หรอกคนให้บริการต้องจัดการเองถ้าจะเปิดผับบาร์
