// In-memory document store for development
// This persists while the server process is running

export interface StoredDocument {
  id: string;
  projectId: string;
  name: string;
  type: string;
  size: number;
  url: string;
  category: string;
  status: 'uploaded' | 'processing' | 'completed' | 'error';
  created_at: string;
  updated_at: string;
  extracted_text?: string;
  confidence?: number;
  error?: string;
}

let documents: StoredDocument[] = [];

export function addDocument(doc: Omit<StoredDocument, 'id' | 'created_at' | 'updated_at'> & { created_at?: string; updated_at?: string }): StoredDocument {
  const now = new Date().toISOString();
  const document: StoredDocument = {
    id: `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    created_at: doc.created_at || now,
    updated_at: doc.updated_at || now,
    ...doc,
  };
  documents.push(document);
  return document;
}

export function getDocumentsByProjectId(projectId: string): StoredDocument[] {
  return documents.filter(doc => doc.projectId === projectId);
}

export function getDocumentById(id: string): StoredDocument | undefined {
  return documents.find(doc => doc.id === id);
}

export function updateDocument(id: string, updates: Partial<StoredDocument>): StoredDocument | undefined {
  const index = documents.findIndex(doc => doc.id === id);
  if (index === -1) return undefined;
  
  const now = new Date().toISOString();
  documents[index] = {
    ...documents[index],
    ...updates,
    updated_at: now,
  };
  
  return documents[index];
}

export function deleteDocument(id: string): boolean {
  const index = documents.findIndex(doc => doc.id === id);
  if (index === -1) return false;
  
  documents.splice(index, 1);
  return true;
}

export function clearDocuments(): void {
  documents = [];
}
