!pip install gdown

import gdown
import os

# أنشئ مجلد لحفظ النماذج إذا لم يكن موجودًا
os.makedirs("models", exist_ok=True)

# قائمة الملفات لتحميلها: {"اسم الملف": "Google Drive File ID"}
models = {
    "Acute_Lymphoblastic_Leukemia.h5": "1q0XtgAb0BAZgCuU23rWAXkKL-p1E2goD",
    "Brain_Tumor_Classification.h5": "1YIncieNKVkvz01tI-QAoXAabbKcEddgm",
    "Breast_Cancer.h5": "1or3mhTyMIfH-Xpr_J_OJDnhikDzc0X61",
    "Cervical_Cancer.h5": "1NiVUwSGan4MUFBJlMDuZ0zjJJ3q3c-WA",
    "Colon_Cancer.h5": "1wF31BV1SqOcljI9p7moTtzLxscTPgGCc",
    "ct_kidney.h5": "1TDqaX2tE8XF7JxYAtnjztU-FOW2eStp_",
    "Lymphoma_Classification.h5": "1ikk1gYPBwTx0FYoLPV0HuPdltwlwViSe",
    "teeth_classification_pretrain.h5": "1FRflfd3QOECd1B6ytfXOseCBFPCLYCOZ",
}

# تنزيل الملفات
for filename, file_id in models.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    output = os.path.join("models", filename)
    print(f"⬇️ Downloading {filename}...")
    gdown.download(url, output, quiet=False)

print("\n✅ All models downloaded to 'models/' folder.")
