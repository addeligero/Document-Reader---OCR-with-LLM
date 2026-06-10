# QuAMS Local Database Integration

This folder replaces the Supabase data/auth/storage pieces with local Flask
routes, SQLAlchemy models, SQLite by default, and local disk uploads.

The project now runs as one Flask backend from the root `app.py`:

- OCR/LLM compatibility endpoint: `POST /upload`
- Database/auth/document API endpoints: `/api/...`
- Document upload endpoint that stores the result: `POST /api/documents/upload`

## System Flow

The frontend now talks to only one backend:

```text
Frontend
  -> Flask backend: app.py
    -> Auth/database API routes
    -> Local file storage in UPLOAD_DIR
    -> OCR in services/document_processor.py
    -> SVM classifier
    -> LLM classifier
    -> SQLite/local database through SQLAlchemy
```

The main document upload flow is:

```text
Frontend uploads PDF/DOCX/image
  -> POST /api/documents/upload
  -> Save original file locally
  -> Run OCR/text extraction
  -> Run SVM category candidate classifier
  -> Run LLM final category/tag classifier
  -> Save document metadata and extracted text to database
  -> Return saved document JSON to frontend
```

Use `POST /api/documents/upload` for the real application workflow. Use
`POST /upload` only when you want OCR/LLM output without saving to the database.

## Run

From the backend root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy database\.env.example .env
python app.py
```

The API runs at `http://localhost:8000` unless `PORT` is set.

## Environment

Example `.env`:

```env
JWT_SECRET=change-this-secret
DATABASE_URL=sqlite:///quams.db
UPLOAD_DIR=uploads
TWO_FACTOR_REQUIRED=false
SESSION_TIMEOUT_MINUTES=30
```

`DATABASE_URL=sqlite:///quams.db` creates a local SQLite database. You can later
point it to another SQLAlchemy-supported database if needed.

## Seeded Login

On first run, the app creates one local admin:

- username: `admin`
- password: `Quams123`

Use `POST /api/auth/login` to get a JWT token. Send that token on protected API
requests as:

```text
Authorization: Bearer <token>
```

## Main API Flow

### 1. Login

```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "Quams123"
}
```

Response:

```json
{
  "token": "jwt_token_here",
  "user": {
    "id": "user-id",
    "username": "admin",
    "role": "admin"
  }
}
```

Save the `token` on the frontend. Send it on protected endpoints with:

```text
Authorization: Bearer <token>
```

### 2. Upload A Document And Save OCR/LLM Results

```http
POST /api/documents/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

file=<pdf/docx/image>
```

This endpoint saves the uploaded file to `UPLOAD_DIR`, runs OCR/LLM locally,
stores `extracted_text`, `primary_category`, `secondary_category`, `tags`, and
returns the saved document row.

Example response shape:

```json
{
  "id": "document-id",
  "created_at": "2026-06-02T10:00:00+00:00",
  "updated_at": "2026-06-02T10:00:00+00:00",
  "user_id": "user-id",
  "file_name": "sample.pdf",
  "primary_category": "Instruction",
  "secondary_category": "Curriculum",
  "tags": ["syllabus", "course", "instruction"],
  "path": "user-id/generated-file-name.pdf",
  "status": "pending",
  "extracted_text": "OCR text here",
  "uploaded_by": "System Admin"
}
```

### 3. Fetch Stored Documents

```http
GET /api/documents
Authorization: Bearer <token>
```

Optional status filter:

```http
GET /api/documents?status=pending
Authorization: Bearer <token>
```

## Endpoints

- `GET /healthz`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/auth/logout`
- `GET /api/documents`
- `POST /api/documents/upload`
- `PATCH /api/documents/<id>`
- `DELETE /api/documents/<id>`
- `GET /api/documents/<id>/download`
- `GET /api/categories`
- `GET /api/settings`
- `PATCH /api/settings`
- `GET /api/users`
- `POST /api/users`
- `PATCH /api/users/<id>`
- `DELETE /api/users/<id>`
- `POST /api/users/<id>/reset-password`
- `PATCH /api/users/<id>/activate`
- `PATCH /api/users/<id>/deactivate`
- `PATCH /api/users/<id>/role`
- `GET /api/notifications`
- `POST /api/notifications`
- `PATCH /api/notifications/<id>/read`
- `DELETE /api/notifications/<id>`
- `GET /api/compliance-items`
- `POST /api/compliance-items`
- `PATCH /api/compliance-items/<id>`
- `PATCH /api/compliance-items/<id>/status`
- `DELETE /api/compliance-items/<id>`
- `GET /api/compliance-accreditations`
- `POST /api/compliance-accreditations`
- `PATCH /api/compliance-accreditations/<name>`
- `DELETE /api/compliance-accreditations/<name>`
- `GET /api/compliance-categories`
- `POST /api/compliance-categories`
- `DELETE /api/compliance-categories/<id>`
- `GET /api/compliance-requirement-categories`
- `POST /api/compliance-requirement-categories`
- `DELETE /api/compliance-requirement-categories`
- `POST /upload` for OCR/LLM only, without saving to the database

## Endpoint Examples

### Health Check

```http
GET /healthz
```

Response:

```json
{
  "status": "ok"
}
```

### Get Current User

```http
GET /api/auth/me
Authorization: Bearer <token>
```

### Update Document

```http
PATCH /api/documents/<document_id>
Authorization: Bearer <token>
Content-Type: application/json

{
  "file_name": "Updated Title.pdf",
  "primary_category": "Research",
  "secondary_category": "Extension",
  "tags": ["proposal", "research"],
  "status": "approved"
}
```

Allowed fields are:

- `file_name`
- `primary_category`
- `secondary_category`
- `tags`
- `status`
- `extracted_text`

### Delete Document

```http
DELETE /api/documents/<document_id>
Authorization: Bearer <token>
```

### Download Document

```http
GET /api/documents/<document_id>/download
Authorization: Bearer <token>
```

### Get Categories

```http
GET /api/categories
Authorization: Bearer <token>
```

Example response:

```json
[
  { "id": 1, "name": "Administration" },
  { "id": 2, "name": "Instruction" },
  { "id": 3, "name": "Research" },
  { "id": 4, "name": "Extension" }
]
```

### List Users (Admin/Coordinator)

```http
GET /api/users
Authorization: Bearer <token>
```

### Create User (Admin/Coordinator)

```http
POST /api/users
Authorization: Bearer <token>
Content-Type: application/json

{
  "username": "jdoe",
  "password": "TempPass123",
  "f_name": "Jane",
  "l_name": "Doe",
  "email": "jdoe@example.com",
  "role": "user",
  "department": "IT",
  "status": true,
  "is_taskforce": false
}
```

### Update User (Admin/Coordinator)

```http
PATCH /api/users/<user_id>
Authorization: Bearer <token>
Content-Type: application/json

{
  "f_name": "Janet",
  "l_name": "Doe",
  "email": "janet.doe@example.com",
  "department": "Compliance",
  "is_taskforce": true
}
```

### Reset User Password (Admin/Coordinator)

```http
POST /api/users/<user_id>/reset-password
Authorization: Bearer <token>
Content-Type: application/json

{
  "password": "NewTempPass456"
}
```

### Activate/Deactivate User (Admin/Coordinator)

```http
PATCH /api/users/<user_id>/activate
Authorization: Bearer <token>
```

```http
PATCH /api/users/<user_id>/deactivate
Authorization: Bearer <token>
```

### Update User Role (Admin/Coordinator)

```http
PATCH /api/users/<user_id>/role
Authorization: Bearer <token>
Content-Type: application/json

{
  "role": "quams_coordinator"
}
```

### Delete User (Admin/Coordinator)

```http
DELETE /api/users/<user_id>
Authorization: Bearer <token>
```

### List Notifications

```http
GET /api/notifications
Authorization: Bearer <token>
```

### Create Notification

```http
POST /api/notifications
Authorization: Bearer <token>
Content-Type: application/json

{
  "user_id": "user-id",
  "title": "Document Ready",
  "message": "Your upload has been processed.",
  "type": "info",
  "link": "/documents/123",
  "metadata": {
    "document_id": "123"
  }
}
```

### Mark Notification As Read

```http
PATCH /api/notifications/<notification_id>/read
Authorization: Bearer <token>
```

### Delete Notification

```http
DELETE /api/notifications/<notification_id>
Authorization: Bearer <token>
```

## Compliance APIs

The compatibility category table is intentionally named `catergories` to match
the existing frontend spelling. Use `/api/compliance-categories` from the
frontend.

### List Compliance Items

```http
GET /api/compliance-items
Authorization: Bearer <token>
```

Response shape:

```json
[
  {
    "id": "uuid",
    "accreditation": "AUN-QA",
    "requirements": ["1.1", "1.2"],
    "remarks": "Needs update",
    "mandatory": ["M1", "M2"],
    "enhancement": ["E1"],
    "status": "pending",
    "created_at": "2026-06-02T10:00:00+00:00",
    "updated_at": "2026-06-02T10:00:00+00:00",
    "supporting_documents": [
      {
        "id": "doc-id",
        "file_name": "sample.pdf",
        "primary_category": "Instruction"
      }
    ]
  }
]
```

### Create Compliance Item

```http
POST /api/compliance-items
Authorization: Bearer <token>
Content-Type: application/json

{
  "accreditation": "AUN-QA",
  "requirements": ["1.1", "1.2"],
  "remarks": "Needs update",
  "mandatory": ["M1", "M2"],
  "enhancement": ["E1"],
  "status": "pending",
  "supporting_documents": ["document-id-1", "document-id-2"]
}
```

`supporting_documents` can also be sent as objects:

```json
{
  "supporting_documents": [
    { "id": "document-id-1" },
    { "document_id": "document-id-2" }
  ]
}
```

### Update Compliance Item

```http
PATCH /api/compliance-items/<item_id>
Authorization: Bearer <token>
Content-Type: application/json

{
  "remarks": "Updated remarks",
  "status": "met",
  "supporting_documents": ["document-id-1"]
}
```

When `supporting_documents`, `document_ids`, or `documents` is included, the
backend replaces the full supporting document list.

### Update Compliance Item Status

```http
PATCH /api/compliance-items/<item_id>/status
Authorization: Bearer <token>
Content-Type: application/json

{
  "status": "not_met"
}
```

Allowed statuses are `met`, `pending`, and `not_met`.

### Delete Compliance Item

```http
DELETE /api/compliance-items/<item_id>
Authorization: Bearer <token>
```

### Compliance Accreditations

```http
GET /api/compliance-accreditations
POST /api/compliance-accreditations
PATCH /api/compliance-accreditations/<name>
DELETE /api/compliance-accreditations/<name>
Authorization: Bearer <token>
```

Create/update payload:

```json
{
  "name": "AUN-QA",
  "requirements": ["1.1", "1.2"]
}
```

Response shape:

```json
{
  "id": "uuid",
  "name": "AUN-QA",
  "requirements": ["1.1", "1.2"],
  "created_at": "2026-06-02T10:00:00+00:00",
  "updated_at": "2026-06-02T10:00:00+00:00"
}
```

### Compliance Categories

```http
GET /api/compliance-categories
POST /api/compliance-categories
DELETE /api/compliance-categories/<id>
Authorization: Bearer <token>
```

Create payload:

```json
{
  "name": "Administration"
}
```

Response shape:

```json
{
  "id": 1,
  "name": "Administration"
}
```

### Requirement-Category Mappings

```http
GET /api/compliance-requirement-categories
Authorization: Bearer <token>
```

Response shape:

```json
[
  {
    "accreditation_name": "AUN-QA",
    "requirement_key": "1",
    "category_id": 2
  }
]
```

Replace mappings for one accreditation and requirement:

```http
POST /api/compliance-requirement-categories
Authorization: Bearer <token>
Content-Type: application/json

{
  "accreditation_name": "AUN-QA",
  "requirement_key": "1",
  "category_ids": [2, 3]
}
```

You can also send one category:

```json
{
  "accreditation_name": "AUN-QA",
  "requirement_key": "1",
  "category_id": 2
}
```

Delete mappings for one accreditation and requirement:

```http
DELETE /api/compliance-requirement-categories
Authorization: Bearer <token>
Content-Type: application/json

{
  "accreditation_name": "AUN-QA",
  "requirement_key": "1"
}
```

### OCR/LLM Only

```http
POST /upload
Content-Type: multipart/form-data

file=<pdf/docx/image>
```

This endpoint returns OCR/classification output but does not create a database
record.

## Frontend Migration From Supabase

Replace direct Supabase calls with Flask API calls:

- Supabase auth sign-in -> `POST /api/auth/login`
- Supabase user/session fetch -> `GET /api/auth/me`
- Supabase documents select -> `GET /api/documents`
- Supabase document insert/upload -> `POST /api/documents/upload`
- Supabase document update -> `PATCH /api/documents/<id>`
- Supabase document delete -> `DELETE /api/documents/<id>`
- Supabase storage download -> `GET /api/documents/<id>/download`
- Supabase compliance items -> `/api/compliance-items`
- Supabase compliance accreditations -> `/api/compliance-accreditations`
- Supabase `catergories` -> `/api/compliance-categories`
- Supabase compliance requirement mappings -> `/api/compliance-requirement-categories`
- Supabase notifications -> `/api/notifications`
- Supabase admin users/profiles -> `/api/users`

For Supabase realtime subscriptions, use polling for now: refresh
`GET /api/documents` every few seconds or immediately after a mutation.

## Frontend Fetch Helper Example

Create one API helper file in the frontend, for example `api.js`:

```js
const API_BASE = "http://localhost:8000";

function authHeaders() {
  const token = localStorage.getItem("token");
  return {
    Authorization: `Bearer ${token}`,
  };
}

export async function login(username, password) {
  const res = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ username, password }),
  });

  const data = await res.json();
  if (data.token) {
    localStorage.setItem("token", data.token);
  }
  return data;
}

export async function getCurrentUser() {
  const res = await fetch(`${API_BASE}/api/auth/me`, {
    headers: authHeaders(),
  });

  return res.json();
}

export async function getDocuments() {
  const res = await fetch(`${API_BASE}/api/documents`, {
    headers: authHeaders(),
  });

  return res.json();
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/documents/upload`, {
    method: "POST",
    headers: authHeaders(),
    body: formData,
  });

  return res.json();
}

export async function updateDocument(id, payload) {
  const res = await fetch(`${API_BASE}/api/documents/${id}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
    },
    body: JSON.stringify(payload),
  });

  return res.json();
}

export async function deleteDocument(id) {
  const res = await fetch(`${API_BASE}/api/documents/${id}`, {
    method: "DELETE",
    headers: authHeaders(),
  });

  return res.json();
}
```

Do not manually set `Content-Type` when sending `FormData`. The browser adds
the correct multipart boundary automatically.

## Frontend Mental Model

```text
Login
  -> Save token
  -> Send token with every /api request
  -> Upload files through /api/documents/upload
  -> Backend handles OCR, LLM, local storage, and database writes
  -> Frontend displays the returned document JSON
```

## Structure

- `app.py` - root single Flask server
- `database/quams_backend/__init__.py` - app factory and database setup
- `database/quams_backend/models.py` - SQLAlchemy tables
- `database/quams_backend/auth.py` - JWT helpers and auth decorators
- `database/quams_backend/routes/` - API blueprints
- `services/document_processor.py` - shared OCR/LLM processing pipeline
