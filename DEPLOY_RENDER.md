# NeuralWiki Render Deployment Guide

## What's Been Prepared

✅ **Dockerfile** - Backend containerization  
✅ **render.yaml** - Infrastructure-as-code (auto-config)  
✅ **requirements.txt** - Python dependencies  
✅ **Environment-aware API URLs** - Frontend auto-connects to backend  
✅ **CORS configuration** - Supports cross-origin requests from frontend  

---

## **Deployment Method A: Using render.yaml (Recommended)**

### **1. Push to GitHub**
```bash
git add Dockerfile render.yaml requirements.txt DEPLOY_RENDER.md
git commit -m "Add Render deployment files"
git push origin HEAD:main
```

### **2. Deploy via Render Dashboard**
1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Blueprint"** (Infrastructure as Code)
3. **Connect GitHub**:
   - Authorize Render
   - Select repository

4. **Select Branch**: `main`
5. **Review Configuration**: render.yaml will auto-detect
6. **Add Secrets** (in Render dashboard → Secrets):
   ```
   CHAT_API_KEY=your-nvidia-api-key-here
   CHAT_API_KEY_FALLBACK=your-nvidia-fallback-key-here
   EXTRACT_API_KEY=your-mistral-api-key-here
   EXTRACT_API_KEY_FALLBACK=your-nvidia-extract-key-here
   WIKI_API_KEY=your-groq-api-key-here
   ```
7. **Deploy**: Click **"Deploy"**

Render will automatically:
- Build both services
- Set environment variables
- Configure CORS automatically
- Link frontend to backend

---

## **Deployment Method B: Manual Setup (If render.yaml doesn't work)**

### **1. Create Backend Service (FastAPI)**

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. **Connect GitHub**:
   - Select your `neuralwiki` repo
   - Authorize Render access

4. **Configure Service**:
   - **Name**: `neuralwiki-backend`
   - **Region**: Oregon (or closest to you)
   - **Branch**: `main`
   - **Runtime**: `Docker`
   - **Dockerfile path**: `./Dockerfile`
   - **Auto-deploy**: ✅ ON

5. **Add Environment Variables** (Render dashboard):
   ```
   CHAT_API_KEY=your-nvidia-api-key-here
   CHAT_API_KEY_FALLBACK=your-nvidia-fallback-key-here
   EXTRACT_API_KEY=your-mistral-api-key-here
   EXTRACT_API_KEY_FALLBACK=your-nvidia-extract-key-here
   WIKI_API_KEY=your-groq-api-key-here
   ```

6. **Deploy**: Click **"Deploy"** (takes 5-10 minutes)
7. **Save URL**: `https://neuralwiki-backend.onrender.com` (appears after deploy)

### **2. Create Frontend Service (React/Vite)**

1. Click **"New +"** → **"Static Site"**
2. **Connect GitHub**:
   - Select same repo

3. **Configure Service**:
   - **Name**: `neuralwiki-frontend`
   - **Branch**: `main`
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Publish directory**: `frontend/dist`

4. **Add Environment Variable**:
   ```
   VITE_API_URL=https://neuralwiki-backend.onrender.com
   ```
   *(Use the backend URL from step 1.7)*

5. **Deploy**: Click **"Deploy"**

---

## **Verify Deployment**

1. **Backend**: Open `https://neuralwiki-backend.onrender.com/docs`
   - Should show FastAPI Swagger UI
   - Try `/` endpoint to verify it's running

2. **Frontend**: Open `https://neuralwiki-frontend.onrender.com`
   - Should load React app
   - Check browser console for API connection logs

3. **Test Connection**:
   - Upload a document in the UI
   - Check backend logs in Render dashboard
   - Verify wiki generation works

---

## **Troubleshooting**

| Issue | Solution |
|-------|----------|
| 502 Bad Gateway | Backend still starting. Wait 5 min, then refresh |
| Build fails | Check logs in Render → service → Logs tab |
| CORS errors | Make sure `FRONTEND_URL` or `VITE_API_URL` match exactly |
| API key errors | Copy-paste without spaces, verify in `.env` locally first |
| "File not found" | Verify `Dockerfile` at repo root (not in backend/) |

---

## **Monitoring & Logs**

1. **Render Dashboard** → Select service
2. **"Logs"** tab shows real-time output
3. **"Metrics"** shows CPU/memory/disk usage
4. **"Events"** shows deploy history

---

## **Auto-Deploy on Push**

Both services automatically redeploy when you push to `main`:

```bash
git add .
git commit -m "Update deployment"
git push origin HEAD:main
```

Render will:
- Pull latest code
- Rebuild Docker image
- Redeploy within 2-5 minutes

---

## **Cost Estimate**

- **Backend (Web Service)**: Free tier = 750 hrs/month
  - *24/7 = ~720 hrs, so ~$7-10/month for overages*
  - *Upgrade to Pro tier ($7/month) for unlimited hours*
- **Frontend (Static)**: **Completely Free**
- **Total**: ~$7-10/month

---

## **Next Steps**

1. ✅ Push code to GitHub
2. ✅ Create Render account
3. ✅ Deploy using Method A or B above
4. ✅ Monitor logs and test
5. ✅ Share your deployed URL!

For help: Check [Render Docs](https://render.com/docs)
