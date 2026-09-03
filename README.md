# Hotel Booking Cancellation — Production

تحويل مشروع النوتبوك (EDA + 8 موديلات + Tuning + Ensembles) لنظام Production
كامل: تدريب مرة واحدة على Kaggle، وAPI + Dashboard بيشتغلوا محليًا/على أي سيرفر
من غير أي حاجة تحتاج GPU.

## الفكرة

```
Kaggle (فيه GPU/موارد)          سيرفرك (مش محتاج GPU)
┌────────────────────┐          ┌───────────────────────────┐
│ train.py            │  ---->  │ artifacts/ (models + json) │
│ + hotel_bookings.csv│  ينزل   │        ↓                   │
└────────────────────┘  المجلد  │  FastAPI (app/main.py)     │
                                 │        ↓                   │
                                 │  Dashboard (static/)        │
                                 └───────────────────────────┘
```

## 1) التدريب على Kaggle

1. ارفع `train.py` كـ Kaggle Notebook أو حطه جنب النوتبوك الحالي.
2. شغّل:
   ```bash
   python train.py --data /kaggle/input/<dataset>/hotel_bookings.csv --out artifacts
   ```
3. لما يخلص، هتلاقي مجلد `artifacts/` فيه:
   - `models/*.pkl` — كل موديل لوحده (Logistic Regression, Decision Tree, Random Forest,
     Gradient Boosting, AdaBoost, KNN, Naive Bayes, XGBoost, XGBoost Tuned F1/Accuracy,
     Soft Voting, Stacking)
   - `preprocessing.pkl` — الـ Encoders/Scalers/Frequency maps
   - `metrics.json` — مقارنة الموديلات مرتبة بالـ F1
   - `feature_importance.json` — أهم 15 عامل
   - `insights.json` — كل أرقام الـ EDA اللي الداشبورد بتعرضها
   - `feature_schema.json` — شكل الأعمدة الخام المطلوبة في `/predict`
4. نزّل مجلد `artifacts/` بالكامل وحطه جنب `Dockerfile` في المشروع ده.

> ⚠️ لازم إصدارات `scikit-learn` / `xgboost` / `imbalanced-learn` في `requirements.txt`
> تكون هي نفسها (أو قريبة) من اللي شغالة على Kaggle، عشان ملفات الـ `.pkl` تتحمّل صح.
> شوف الإصدارات على Kaggle بـ `pip show scikit-learn xgboost imbalanced-learn` وحدّث
> `requirements.txt` لو مختلفة.

## 2) التشغيل محليًا (بدون Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8000/dashboard

## 3) Docker

```bash
docker compose up --build
```

نفس الروابط فوق. الـ `artifacts/` بتتحمّل كـ volume (read-only) عشان تقدر تحدّث
الموديلات من غير ما تعيد بناء الـ image.

## 4) اختبار بـ Postman

استورد `postman_collection.json` في Postman، وشغّل الـ requests بالترتيب:
1. `Health Check` — لازم يرجع `models_loaded` فيها كل الموديلات
2. `List Models` — تأكد إن الترتيب صح (الأعلى F1 الأول)
3. `Feature Importance` / `Insights` / `Raw Feature Schema`
4. `Predict` — جرب بموديل حقيقي من نتيجة `/models` (استخدم قيمة `file` من غير `.pkl`)
5. `Predict - Invalid Model` — لازم يرجع 404

بعد ما كل الـ requests تنجح، يبقى جاهز ترفعه.

## Endpoints

| Method | Path        | الوصف                                    |
|--------|-------------|-------------------------------------------|
| GET    | `/health`   | حالة الـ API والموديلات المحمّلة           |
| GET    | `/models`   | كل الموديلات مرتبة (الأحسن للأسوأ بالـ F1) |
| GET    | `/features` | أهم 15 Feature                            |
| GET    | `/insights` | كل أرقام/جداول الـ EDA                     |
| GET    | `/schema`   | شكل بيانات الحجز الخام المطلوبة            |
| POST   | `/predict`  | `{ "model_key": "...", "booking": {...} }` |

## هيكل المشروع

```
hotel_cancellation_mlops/
├── train.py                 # يتشغل مرة واحدة على Kaggle
├── app/
│   └── main.py               # FastAPI
├── static/                   # الداشبورد (HTML/CSS/JS)
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── config.js
├── artifacts/                 # (تتنزل من Kaggle) الموديلات + الـ JSON files
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── postman_collection.json
```

## 5) النشر على Railway / Render

المنصتين بيبنوا الـ Docker image مباشرة من الـ `Dockerfile` وبيوفروا HTTPS ودومين
مجاني تلقائي (`*.up.railway.app` أو `*.onrender.com`) — مش محتاج دومين خاص علشان تبدأ.

**Render:**
1. اعمل New → Web Service → اربط الـ repo.
2. Render هياخد `render.yaml` تلقائي (Environment = Docker).
3. لو مش عايز تستخدمه، اختار يدويًا: Environment=Docker, Health Check Path=`/health`.
4. الأهم: **لازم `artifacts/` تكون داخل الـ repo قبل الـ push** (الملفات اللي نزلتها
   من Kaggle) لأن الـ Dockerfile بيعمل `COPY artifacts ./artifacts` وقت الـ build.

**Railway:**
1. New Project → Deploy from GitHub repo.
2. Railway بيكتشف الـ `Dockerfile` تلقائي ويستخدمه.
3. Railway بيحقن متغير `PORT` تلقائي — الـ Dockerfile متظبط عليه خلاص
   (`--port ${PORT}`).
4. لو حابب تقفل CORS بعدين، ضيف Variable باسم `ALLOWED_ORIGINS` بقيمة
   دومين الفرونت اند (مفصولة بفاصلة لو أكتر من واحد).

**بعد النشر:**
- API: `https://<your-app>.onrender.com` أو `.up.railway.app`
- Dashboard: نفس الدومين + `/dashboard`
- جرب Postman collection تاني بس غيّر `base_url` variable للدومين الجديد بدل
  `localhost:8000`.

> ملحوظة حجم: موديلات زي Random Forest / Stacking ممكن تبقى كبيرة نسبيًا (عشرات
> الـ MB). لو الـ repo أو الـ image بقى كبير جدًا، فكّر تشيل الموديلات الأضعف أداءً
> من `artifacts/models/` قبل الرفع وسيب بس أفضل 4-5 موديلات — الداشبورد والـ API
> هيشتغلوا عادي بنفس المنطق.

## ملاحظات قبل الرفع

- الحقول اللي الفورم بتبعتها لازم تكون مطابقة لأعمدة `hotel_bookings.csv` الخام
  (شوف `feature_schema.json` لو عندك أعمدة إضافية أو أسامي مختلفة).
- لو عايز تضيف موديل جديد بعدين، كفاية تحط ملف `.pkl` جديد في `artifacts/models/`
  وتضيف صف ليه في `metrics.json` — الـ API والداشبورد هيلقطوه تلقائي.
- الـ CORS مفتوح `*` دلوقتي للتجربة، لما تحدد الـ Frontend domain بتاع الـ Production
  قفّله في `app/main.py`.
