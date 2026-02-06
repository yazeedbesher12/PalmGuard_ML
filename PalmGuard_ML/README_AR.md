# PalmGuard ML (ملخص عربي)

PalmGuard مشروع تجريبي لاكتشاف سوسة النخيل الحمراء مبكرًا من خلال تسجيلات صوتية للأشجار.

## ماذا يفعل؟
- يحوّل ملفات WAV إلى خصائص log-mel
- يدرب نموذج CNN بسيط لتصنيف (سليم/مصاب)
- يجمع نتائج المقاطع ليخرج **درجة خطورة للشجرة**
- يوفر واجهة Streamlit للتجربة (تدريب/استدلال/عرض النتائج)

## تشغيل سريع
```bash
pip install -r requirements.txt
streamlit run app.py
```

## التدريب والاستدلال
```bash
python train.py
python infer.py
```

راجع `README.md` للتفاصيل الكاملة، و `QUALITY_REPORT.md` لتقرير الجودة.
