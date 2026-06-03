-- ============================================================
-- EXTENSIONS
-- ============================================================
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================
-- PROFILES
-- ============================================================
CREATE TABLE IF NOT EXISTS public.profiles (
  id uuid NOT NULL,
  f_name text NOT NULL,
  l_name text NOT NULL,
  role text NOT NULL,
  department text NULL,
  status boolean NULL DEFAULT true,
  created_at timestamptz NULL DEFAULT now(),
  updated_at timestamptz NULL DEFAULT now(),
  extension text NULL,
  username text NULL,
  email text NULL,
  avatar text NULL,
  is_taskforce boolean NOT NULL DEFAULT false,
  CONSTRAINT profiles_pkey PRIMARY KEY (id),
  CONSTRAINT profiles_avatar_key UNIQUE (avatar),
  CONSTRAINT profiles_username_key UNIQUE (username),
  CONSTRAINT profiles_id_fkey
    FOREIGN KEY (id) REFERENCES auth.users (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_profiles_username
  ON public.profiles USING btree (username);

-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS set_profiles_updated_at ON public.profiles;
CREATE TRIGGER set_profiles_updated_at
BEFORE UPDATE ON public.profiles
FOR EACH ROW
EXECUTE FUNCTION public.handle_updated_at();

CREATE OR REPLACE FUNCTION public.is_privileged_user()
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.profiles p
    WHERE p.id = auth.uid()
      AND lower(p.role) IN (
        'admin',
        'dean',
        'associate_dean',
        'quams_coordinator',
        'department'
      )
      AND COALESCE(p.status, true) = true
  );
$$;

CREATE OR REPLACE FUNCTION public.can_manage_compliance_matrix()
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT (
    public.is_privileged_user()
    OR EXISTS (
      SELECT 1
      FROM public.profiles p
      WHERE p.id = auth.uid()
        AND COALESCE(p.status, true) = true
        AND COALESCE(p.is_taskforce, false) = true
    )
  );
$$;

CREATE OR REPLACE FUNCTION public.can_manage_compliance_accreditations()
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.profiles p
    WHERE p.id = auth.uid()
      AND COALESCE(p.status, true) = true
      AND lower(p.role) = 'quams_coordinator'
  );
$$;

-- ============================================================
-- DOCUMENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  user_id uuid NOT NULL,
  file_name text NOT NULL,
  primary_category text,
  secondary_category text,
  tags text[] NOT NULL DEFAULT '{}',
  path text NOT NULL,
  status text NOT NULL DEFAULT 'pending',
  CONSTRAINT tags_len_check CHECK (array_length(tags, 1) BETWEEN 3 AND 5),
  CONSTRAINT path_unique UNIQUE (path),
  CONSTRAINT documents_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES public.profiles(id) ON DELETE CASCADE
);

DROP TRIGGER IF EXISTS set_documents_updated_at ON public.documents;
CREATE TRIGGER set_documents_updated_at
BEFORE UPDATE ON public.documents
FOR EACH ROW
EXECUTE FUNCTION public.handle_updated_at();

ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "documents_select_all_authenticated" ON public.documents;
CREATE POLICY "documents_select_all_authenticated"
ON public.documents
FOR SELECT
TO authenticated
USING (true);

DROP POLICY IF EXISTS "documents_insert_authenticated" ON public.documents;
CREATE POLICY "documents_insert_authenticated"
ON public.documents
FOR INSERT
TO authenticated
WITH CHECK (user_id = auth.uid());

DROP POLICY IF EXISTS "documents_update_privileged_only" ON public.documents;
CREATE POLICY "documents_update_privileged_only"
ON public.documents
FOR UPDATE
TO authenticated
USING (public.is_privileged_user())
WITH CHECK (public.is_privileged_user());

DROP POLICY IF EXISTS "documents_delete_privileged_only" ON public.documents;
CREATE POLICY "documents_delete_privileged_only"
ON public.documents
FOR DELETE
TO authenticated
USING (public.is_privileged_user());

DROP POLICY IF EXISTS "documents_update_owner" ON public.documents;
CREATE POLICY "documents_update_owner"
ON public.documents
FOR UPDATE
TO authenticated
USING (user_id = auth.uid())
WITH CHECK (user_id = auth.uid());

-- ============================================================
-- NOTIFICATIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.notifications (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at timestamptz DEFAULT now(),
  user_id uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  title text NOT NULL,
  message text NOT NULL,
  type text NOT NULL CHECK (type IN ('info', 'success', 'warning', 'error')),
  read boolean DEFAULT false,
  link text,
  metadata jsonb
);

ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own notifications" ON public.notifications;
CREATE POLICY "Users can view own notifications"
ON public.notifications
FOR SELECT
USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own notifications" ON public.notifications;
CREATE POLICY "Users can update own notifications"
ON public.notifications
FOR UPDATE
USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Service can insert notifications" ON public.notifications;
CREATE POLICY "Service can insert notifications"
ON public.notifications
FOR INSERT
WITH CHECK (true);

DROP POLICY IF EXISTS "Users can delete own notifications" ON public.notifications;
CREATE POLICY "Users can delete own notifications"
ON public.notifications
FOR DELETE
USING (auth.uid() = user_id);

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM pg_publication
    WHERE pubname = 'supabase_realtime'
  ) THEN
    BEGIN
      ALTER PUBLICATION supabase_realtime ADD TABLE public.notifications;
    EXCEPTION
      WHEN duplicate_object THEN NULL;
      WHEN others THEN NULL;
    END;
  END IF;
END;
$$;

-- ============================================================
-- DOCUMENT NOTIFICATION FUNCTIONS / TRIGGERS
-- ============================================================
CREATE OR REPLACE FUNCTION public.notify_on_document_status_change()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  IF NEW.status = 'approved' AND OLD.status IS DISTINCT FROM 'approved' THEN
    INSERT INTO public.notifications (user_id, title, message, type, link, metadata)
    VALUES (
      NEW.user_id,
      'Document Approved',
      'Your document "' || NEW.file_name || '" has been approved.',
      'success',
      '/dashboard/repository',
      jsonb_build_object('document_id', NEW.id, 'file_name', NEW.file_name)
    );

  ELSIF NEW.status = 'pending' AND OLD.status IS DISTINCT FROM 'pending' THEN
    INSERT INTO public.notifications (user_id, title, message, type, link, metadata)
    VALUES (
      NEW.user_id,
      'Document Processed',
      'Your document "' || NEW.file_name || '" has been processed and is now pending admin review.',
      'info',
      '/dashboard/upload',
      jsonb_build_object('document_id', NEW.id, 'file_name', NEW.file_name)
    );

  ELSIF NEW.status = 'rejected' AND OLD.status IS DISTINCT FROM 'rejected' THEN
    INSERT INTO public.notifications (user_id, title, message, type, link, metadata)
    VALUES (
      NEW.user_id,
      'Document Rejected',
      'Your document "' || NEW.file_name || '" has been rejected.',
      'error',
      '/dashboard/upload',
      jsonb_build_object('document_id', NEW.id, 'file_name', NEW.file_name)
    );
  END IF;

  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_document_status_change ON public.documents;
CREATE TRIGGER on_document_status_change
AFTER UPDATE OF status ON public.documents
FOR EACH ROW
EXECUTE FUNCTION public.notify_on_document_status_change();

CREATE OR REPLACE FUNCTION public.notify_validators_on_pending_document()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  IF NEW.status = 'pending'
     AND (TG_OP = 'INSERT' OR COALESCE(OLD.status, '') <> 'pending') THEN
    INSERT INTO public.notifications (user_id, title, message, type, link, metadata)
    SELECT
      p.id,
      'New Document Pending Review',
      'A new document "' || NEW.file_name || '" needs your review.',
      'warning',
      '/dashboard/classification',
      jsonb_build_object('document_id', NEW.id, 'file_name', NEW.file_name)
    FROM public.profiles p
    WHERE lower(p.role) IN ('admin', 'dean', 'quams_coordinator', 'associate_dean', 'department')
      AND COALESCE(p.status, true) = true;
  END IF;

  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_document_pending_review ON public.documents;
CREATE TRIGGER on_document_pending_review
AFTER INSERT OR UPDATE OF status ON public.documents
FOR EACH ROW
EXECUTE FUNCTION public.notify_validators_on_pending_document();

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM pg_publication
    WHERE pubname = 'supabase_realtime'
  ) THEN
    BEGIN
      ALTER PUBLICATION supabase_realtime ADD TABLE public.documents;
    EXCEPTION
      WHEN duplicate_object THEN NULL;
      WHEN others THEN NULL;
    END;
  END IF;
END;
$$;

-- ============================================================
-- PROFILE RLS
-- ============================================================
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Privileged users can read all profiles" ON public.profiles;
CREATE POLICY "Privileged users can read all profiles"
ON public.profiles
FOR SELECT
TO authenticated
USING (public.is_privileged_user());

DROP POLICY IF EXISTS "Privileged users can update all profiles" ON public.profiles;
CREATE POLICY "Privileged users can update all profiles"
ON public.profiles
FOR UPDATE
TO authenticated
USING (public.is_privileged_user())
WITH CHECK (public.is_privileged_user());

DROP POLICY IF EXISTS "Authenticated users can read profiles for attribution" ON public.profiles;
CREATE POLICY "Authenticated users can read profiles for attribution"
ON public.profiles
FOR SELECT
TO authenticated
USING (true);

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM pg_publication
    WHERE pubname = 'supabase_realtime'
  ) THEN
    BEGIN
      ALTER PUBLICATION supabase_realtime ADD TABLE public.profiles;
    EXCEPTION
      WHEN duplicate_object THEN NULL;
      WHEN others THEN NULL;
    END;
  END IF;
END;
$$;

-- ============================================================
-- APP SETTINGS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.app_settings (
  key text PRIMARY KEY,
  value text NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.app_settings ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Public read app settings" ON public.app_settings;
CREATE POLICY "Public read app settings"
ON public.app_settings
FOR SELECT
USING (true);

INSERT INTO public.app_settings (key, value)
VALUES ('two_factor_required', 'false')
ON CONFLICT (key) DO NOTHING;

-- ============================================================
-- COMPLIANCE ITEMS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.compliance_items (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  accreditation text NOT NULL,
  requirements text[] NOT NULL DEFAULT '{}',
  remarks text NOT NULL DEFAULT '',
  mandatory text[] NOT NULL DEFAULT '{}',
  enhancement text[] NOT NULL DEFAULT '{}',
  status text NOT NULL DEFAULT 'pending'
);

DROP TRIGGER IF EXISTS set_compliance_updated_at ON public.compliance_items;
CREATE TRIGGER set_compliance_updated_at
BEFORE UPDATE ON public.compliance_items
FOR EACH ROW
EXECUTE FUNCTION public.handle_updated_at();

ALTER TABLE public.compliance_items ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "compliance_items_select" ON public.compliance_items;
CREATE POLICY "compliance_items_select"
ON public.compliance_items
FOR SELECT
TO authenticated
USING (true);

DROP POLICY IF EXISTS "compliance_items_insert" ON public.compliance_items;
CREATE POLICY "compliance_items_insert"
ON public.compliance_items
FOR INSERT
TO authenticated
WITH CHECK (public.can_manage_compliance_matrix());

DROP POLICY IF EXISTS "compliance_items_update" ON public.compliance_items;
CREATE POLICY "compliance_items_update"
ON public.compliance_items
FOR UPDATE
TO authenticated
USING (public.can_manage_compliance_matrix())
WITH CHECK (public.can_manage_compliance_matrix());

DROP POLICY IF EXISTS "compliance_items_delete" ON public.compliance_items;
CREATE POLICY "compliance_items_delete"
ON public.compliance_items
FOR DELETE
TO authenticated
USING (public.can_manage_compliance_matrix());

-- ============================================================
-- COMPLIANCE ITEM DOCUMENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.compliance_item_documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  compliance_item_id uuid NOT NULL REFERENCES public.compliance_items(id) ON DELETE CASCADE,
  document_id uuid NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
  UNIQUE (compliance_item_id, document_id)
);

ALTER TABLE public.compliance_item_documents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "compliance_item_docs_select" ON public.compliance_item_documents;
CREATE POLICY "compliance_item_docs_select"
ON public.compliance_item_documents
FOR SELECT
TO authenticated
USING (true);

DROP POLICY IF EXISTS "compliance_item_docs_insert" ON public.compliance_item_documents;
CREATE POLICY "compliance_item_docs_insert"
ON public.compliance_item_documents
FOR INSERT
TO authenticated
WITH CHECK (public.can_manage_compliance_matrix());

DROP POLICY IF EXISTS "compliance_item_docs_delete" ON public.compliance_item_documents;
CREATE POLICY "compliance_item_docs_delete"
ON public.compliance_item_documents
FOR DELETE
TO authenticated
USING (public.can_manage_compliance_matrix());

-- ============================================================
-- COMPLIANCE VIEW
-- ============================================================
CREATE OR REPLACE VIEW public.compliance_documents
WITH (security_invoker = true) AS
SELECT
  cid.compliance_item_id,
  cid.id,
  d.id AS document_id,
  d.file_name,
  d.primary_category
FROM public.compliance_item_documents cid
JOIN public.documents d
  ON d.id = cid.document_id;

REVOKE ALL ON public.compliance_documents FROM anon;
GRANT SELECT ON public.compliance_documents TO authenticated;

-- ============================================================
-- COMPLIANCE ACCREDITATIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.compliance_accreditations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  name text NOT NULL UNIQUE,
  requirements text[] NOT NULL DEFAULT '{}'
);

DROP TRIGGER IF EXISTS set_compliance_accreditations_updated_at ON public.compliance_accreditations;
CREATE TRIGGER set_compliance_accreditations_updated_at
BEFORE UPDATE ON public.compliance_accreditations
FOR EACH ROW
EXECUTE FUNCTION public.handle_updated_at();

ALTER TABLE public.compliance_accreditations ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "compliance_accreditations_select" ON public.compliance_accreditations;
CREATE POLICY "compliance_accreditations_select"
ON public.compliance_accreditations
FOR SELECT
TO authenticated
USING (true);

DROP POLICY IF EXISTS "compliance_accreditations_insert" ON public.compliance_accreditations;
CREATE POLICY "compliance_accreditations_insert"
ON public.compliance_accreditations
FOR INSERT
TO authenticated
WITH CHECK (public.can_manage_compliance_accreditations());

DROP POLICY IF EXISTS "compliance_accreditations_update" ON public.compliance_accreditations;
CREATE POLICY "compliance_accreditations_update"
ON public.compliance_accreditations
FOR UPDATE
TO authenticated
USING (public.can_manage_compliance_accreditations())
WITH CHECK (public.can_manage_compliance_accreditations());

DROP POLICY IF EXISTS "compliance_accreditations_delete" ON public.compliance_accreditations;
CREATE POLICY "compliance_accreditations_delete"
ON public.compliance_accreditations
FOR DELETE
TO authenticated
USING (public.can_manage_compliance_accreditations());

INSERT INTO public.compliance_accreditations (name, requirements)
VALUES
  (
    'AACCUP',
    ARRAY[
      'Area 1 - Vision, Mission, Goals and Objectives',
      'Area 2 - Faculty',
      'Area 3 - Curriculum and Instruction',
      'Area 4 - Support to Students',
      'Area 5 - Research',
      'Area 6 - Extension and Community Involvement',
      'Area 7 - Library',
      'Area 8 - Physical Plant and Facilities',
      'Area 9 - Laboratories',
      'Area 10 - Administration'
    ]
  ),
  (
    'PICAB',
    ARRAY[
      '1.0 Background Information',
      '2.0 Institutional Summary',
      '3.0 Program Educational Objectives',
      '4.0 Program Outcomes (Student Outcomes)',
      '5.0 Curriculum',
      '6.0 Students',
      '7.0 Faculty',
      '8.0 Facilities',
      '9.0 Institutional Support',
      '10.0 Industry-Academe Linkage and Community Programs',
      '11.0 Program Improvement'
    ]
  ),
  (
    'COE',
    ARRAY[
      'Criterion 1 - Innovation Culture',
      'Criterion 2 - Staff Development Tradition',
      'Criterion 3 - Learner and Graduate Quality',
      'Criterion 4 - Culture of Research and Creativity',
      'Criterion 5 - International Outlook',
      'Criterion 6 - Service Orientation'
    ]
  ),
  (
    'AUN-QA',
    ARRAY[
      'Criterion 1 - University Information',
      'Criterion 2 - Programme Structure and Content',
      'Criterion 3 - Teaching and Learning Approach',
      'Criterion 4 - Academic Staff',
      'Criterion 5 - Academic Staff Support',
      'Criterion 6 - Student Support Services',
      'Criterion 7 - Facilities and Infrastructure',
      'Criterion 8 - Output and Outcomes'
    ]
  ),
  ('ISO', ARRAY[]::text[])
ON CONFLICT (name) DO NOTHING;

-- ============================================================
-- CATEGORIES
-- NOTE: you originally used "catergories"
-- kept here to match your existing schema
-- ============================================================
CREATE TABLE IF NOT EXISTS public.catergories (
  id integer PRIMARY KEY,
  name text NOT NULL UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

DROP TRIGGER IF EXISTS set_catergories_updated_at ON public.catergories;
CREATE TRIGGER set_catergories_updated_at
BEFORE UPDATE ON public.catergories
FOR EACH ROW
EXECUTE FUNCTION public.handle_updated_at();

ALTER TABLE public.catergories ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "catergories_select" ON public.catergories;
CREATE POLICY "catergories_select"
ON public.catergories
FOR SELECT
TO authenticated
USING (true);

DROP POLICY IF EXISTS "catergories_insert" ON public.catergories;
CREATE POLICY "catergories_insert"
ON public.catergories
FOR INSERT
TO authenticated
WITH CHECK (public.can_manage_compliance_accreditations());

DROP POLICY IF EXISTS "catergories_update" ON public.catergories;
CREATE POLICY "catergories_update"
ON public.catergories
FOR UPDATE
TO authenticated
USING (public.can_manage_compliance_accreditations())
WITH CHECK (public.can_manage_compliance_accreditations());

DROP POLICY IF EXISTS "catergories_delete" ON public.catergories;
CREATE POLICY "catergories_delete"
ON public.catergories
FOR DELETE
TO authenticated
USING (public.can_manage_compliance_accreditations());

-- ============================================================
-- COMPLIANCE REQUIREMENT CATEGORIES
-- ============================================================
CREATE TABLE IF NOT EXISTS public.compliance_requirement_categories (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  accreditation_name text NOT NULL,
  requirement_key text NOT NULL,
  category_id integer NOT NULL REFERENCES public.catergories(id) ON DELETE CASCADE,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (accreditation_name, requirement_key, category_id),
  CONSTRAINT compliance_requirement_categories_accreditation_fk
    FOREIGN KEY (accreditation_name)
    REFERENCES public.compliance_accreditations(name)
    ON UPDATE CASCADE
    ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_compliance_req_categories_lookup
  ON public.compliance_requirement_categories (accreditation_name, requirement_key);

ALTER TABLE public.compliance_requirement_categories ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "compliance_requirement_categories_select" ON public.compliance_requirement_categories;
CREATE POLICY "compliance_requirement_categories_select"
ON public.compliance_requirement_categories
FOR SELECT
TO authenticated
USING (true);

DROP POLICY IF EXISTS "compliance_requirement_categories_insert" ON public.compliance_requirement_categories;
CREATE POLICY "compliance_requirement_categories_insert"
ON public.compliance_requirement_categories
FOR INSERT
TO authenticated
WITH CHECK (public.can_manage_compliance_accreditations());

DROP POLICY IF EXISTS "compliance_requirement_categories_update" ON public.compliance_requirement_categories;
CREATE POLICY "compliance_requirement_categories_update"
ON public.compliance_requirement_categories
FOR UPDATE
TO authenticated
USING (public.can_manage_compliance_accreditations())
WITH CHECK (public.can_manage_compliance_accreditations());

DROP POLICY IF EXISTS "compliance_requirement_categories_delete" ON public.compliance_requirement_categories;
CREATE POLICY "compliance_requirement_categories_delete"
ON public.compliance_requirement_categories
FOR DELETE
TO authenticated
USING (public.can_manage_compliance_accreditations());

-- ============================================================
-- STORAGE POLICIES (SUPABASE-SPECIFIC)
-- ============================================================

DROP POLICY IF EXISTS "storage_select_all_authenticated_documents" ON storage.objects;
CREATE POLICY "storage_select_all_authenticated_documents"
ON storage.objects
FOR SELECT
TO authenticated
USING (bucket_id = 'documents');

DROP POLICY IF EXISTS "storage_insert_authenticated_documents" ON storage.objects;
CREATE POLICY "storage_insert_authenticated_documents"
ON storage.objects
FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'documents');

DROP POLICY IF EXISTS "storage_delete_privileged_only_documents" ON storage.objects;
CREATE POLICY "storage_delete_privileged_only_documents"
ON storage.objects
FOR DELETE
TO authenticated
USING (
  bucket_id = 'documents'
  AND public.is_privileged_user()
);

DROP POLICY IF EXISTS "Users can update their own avatar" ON storage.objects;
CREATE POLICY "Users can update their own avatar"
ON storage.objects
FOR UPDATE
TO authenticated
USING (
  bucket_id = 'avatars'
  AND auth.uid()::text = (storage.foldername(name))[1]
);

DROP POLICY IF EXISTS "Users can upload their own avatar" ON storage.objects;
CREATE POLICY "Users can upload their own avatar"
ON storage.objects
FOR INSERT
TO authenticated
WITH CHECK (
  bucket_id = 'avatars'
  AND auth.uid()::text = (storage.foldername(name))[1]
);

DROP POLICY IF EXISTS "Users can read their own avatar" ON storage.objects;
CREATE POLICY "Users can read their own avatar"
ON storage.objects
FOR SELECT
TO authenticated
USING (
  bucket_id = 'avatars'
  AND auth.uid()::text = (storage.foldername(name))[1]
);