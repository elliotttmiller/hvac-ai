import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';

// --- Types ---
export interface Document {
  id: string;
  projectId: string;
  name: string;
  url: string;
  type: string;
  size: number;
  uploadedAt: string;
  status: 'processing' | 'completed' | 'error';
  extractedText?: string;
  confidence?: number;
  analysisId?: string;
}

export interface Project {
  id: string;
  name: string;
  location?: string;
  createdAt: string;
  status: 'active' | 'archived';
}

interface DatabaseSchema {
  projects: Project[];
  documents: Document[];
}

// --- DB Logic ---
const DB_PATH = path.join(process.cwd(), 'local-database.json');

class LocalDatabase {
  private data: DatabaseSchema;

  constructor() {
    this.data = this.load();
  }

  private load(): DatabaseSchema {
    try {
      if (!fs.existsSync(DB_PATH)) {
        // Initialize empty DB
        const initial: DatabaseSchema = { projects: [], documents: [] };
        fs.writeFileSync(DB_PATH, JSON.stringify(initial, null, 2));
        return initial;
      }
      const fileContent = fs.readFileSync(DB_PATH, 'utf-8');
      const parsed = JSON.parse(fileContent);
      // Ensure both collections exist
      return {
        projects: parsed.projects || [],
        documents: parsed.documents || [],
      };
    } catch (error) {
      console.error('DB Load Error:', error);
      return { projects: [], documents: [] };
    }
  }

  private save() {
    try {
      fs.writeFileSync(DB_PATH, JSON.stringify(this.data, null, 2));
    } catch (error) {
      console.error('DB Save Error:', error);
    }
  }

  // --- Projects ---

  public getProjects(): Project[] {
    this.data = this.load();
    return this.data.projects;
  }

  public getProjectById(id: string): Project | undefined {
    this.data = this.load();
    return this.data.projects.find((p) => p.id === id);
  }

  public createProject(
    name: string,
    location?: string,
  ): Project {
    const newProject: Project = {
      id: uuidv4(),
      name,
      location: location || 'Unknown',
      createdAt: new Date().toISOString(),
      status: 'active',
    };
    this.data.projects.unshift(newProject);
    this.save();
    return newProject;
  }

  public deleteProject(id: string): boolean {
    const projectIndex = this.data.projects.findIndex((p) => p.id === id);
    if (projectIndex === -1) return false;

    this.data.projects.splice(projectIndex, 1);
    // Also delete associated documents
    this.data.documents = this.data.documents.filter(
      (doc) => doc.projectId !== id,
    );
    this.save();
    return true;
  }

  // --- Documents ---

  public getDocumentsByProjectId(projectId: string): Document[] {
    this.data = this.load();
    return this.data.documents.filter((doc) => doc.projectId === projectId);
  }

  public getDocumentById(id: string): Document | undefined {
    this.data = this.load();
    return this.data.documents.find((doc) => doc.id === id);
  }

  public createDocument(
    projectId: string,
    fileInfo: Omit<
      Document,
      'id' | 'uploadedAt' | 'projectId' | 'status'
    >,
  ): Document {
    const newDoc: Document = {
      id: uuidv4(),
      projectId,
      uploadedAt: new Date().toISOString(),
      status: 'processing',
      ...fileInfo,
    };
    this.data.documents.push(newDoc);
    this.save();
    return newDoc;
  }

  public updateDocument(id: string, updates: Partial<Document>): Document | undefined {
    const index = this.data.documents.findIndex((doc) => doc.id === id);
    if (index === -1) return undefined;

    this.data.documents[index] = {
      ...this.data.documents[index],
      ...updates,
    };
    this.save();
    return this.data.documents[index];
  }

  public deleteDocument(id: string): boolean {
    const index = this.data.documents.findIndex((doc) => doc.id === id);
    if (index === -1) return false;

    this.data.documents.splice(index, 1);
    this.save();
    return true;
  }
}

// Singleton instance
export const db = new LocalDatabase();
