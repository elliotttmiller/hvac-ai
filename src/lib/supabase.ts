import { createClient } from '@supabase/supabase-js';
import { createClientComponentClient, createServerComponentClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

if (!supabaseUrl || !supabaseAnonKey) {
  console.warn('Supabase environment variables not found. Using fallback configuration.');
}

// Client-side Supabase client
export const supabase = createClient(
  supabaseUrl || 'https://placeholder.supabase.co',
  supabaseAnonKey || 'placeholder-key',
  {
    auth: {
      persistSession: true,
      autoRefreshToken: true,
    },
    realtime: {
      params: {
        eventsPerSecond: 10,
      },
    },
  }
);

// Server-side Supabase client
export const createServerSupabaseClient = () => {
  return createServerComponentClient({ cookies });
};

// Client component Supabase client
export const createClientSupabaseClient = () => {
  return createClientComponentClient();
};

// Service role client for admin operations
export const supabaseAdmin = createClient(
  supabaseUrl || 'https://placeholder.supabase.co',
  supabaseServiceKey || 'placeholder-service-key',
  {
    auth: {
      autoRefreshToken: false,
      persistSession: false
    }
  }
);

// Database types
export interface Database {
  public: {
    Tables: {
      users: {
        Row: {
          id: string;
          email: string;
          name: string;
          role: string;
          department: string;
          avatar_url?: string;
          phone?: string;
          location?: string;
          permissions: string[];
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          email: string;
          name: string;
          role: string;
          department: string;
          avatar_url?: string;
          phone?: string;
          location?: string;
          permissions?: string[];
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          email?: string;
          name?: string;
          role?: string;
          department?: string;
          avatar_url?: string;
          phone?: string;
          location?: string;
          permissions?: string[];
          updated_at?: string;
        };
      };
      projects: {
        Row: {
          id: string;
          name: string;
          description: string;
          status: 'planning' | 'design' | 'construction' | 'completed';
          progress: number;
          start_date: string;
          end_date: string;
          budget: number;
          spent: number;
          location: string;
          phase: string;
          created_by: string;
          team_members: string[];
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          name: string;
          description: string;
          status?: 'planning' | 'design' | 'construction' | 'completed';
          progress?: number;
          start_date: string;
          end_date: string;
          budget: number;
          spent?: number;
          location: string;
          phase: string;
          created_by: string;
          team_members?: string[];
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          name?: string;
          description?: string;
          status?: 'planning' | 'design' | 'construction' | 'completed';
          progress?: number;
          start_date?: string;
          end_date?: string;
          budget?: number;
          spent?: number;
          location?: string;
          phase?: string;
          team_members?: string[];
          updated_at?: string;
        };
      };
      documents: {
        Row: {
          id: string;
          name: string;
          type: string;
          status: 'uploaded' | 'processing' | 'completed' | 'error';
          size: number;
          url: string;
          project_id: string;
          uploaded_by: string;
          category?: string;
          extracted_text?: string;
          confidence?: number;
          metadata?: Record<string, any>;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          name: string;
          type: string;
          status?: 'uploaded' | 'processing' | 'completed' | 'error';
          size: number;
          url: string;
          project_id: string;
          uploaded_by: string;
          category?: string;
          extracted_text?: string;
          confidence?: number;
          metadata?: Record<string, any>;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          name?: string;
          type?: string;
          status?: 'uploaded' | 'processing' | 'completed' | 'error';
          size?: number;
          url?: string;
          project_id?: string;
          category?: string;
          extracted_text?: string;
          confidence?: number;
          metadata?: Record<string, any>;
          updated_at?: string;
        };
      };
      chat_messages: {
        Row: {
          id: string;
          content: string;
          role: 'user' | 'assistant';
          agent_type?: string;
          user_id: string;
          project_id?: string;
          metadata?: Record<string, any>;
          created_at: string;
        };
        Insert: {
          id?: string;
          content: string;
          role: 'user' | 'assistant';
          agent_type?: string;
          user_id: string;
          project_id?: string;
          metadata?: Record<string, any>;
          created_at?: string;
        };
        Update: {
          id?: string;
          content?: string;
          role?: 'user' | 'assistant';
          agent_type?: string;
          user_id?: string;
          project_id?: string;
          metadata?: Record<string, any>;
        };
      };
      tasks: {
        Row: {
          id: string;
          title: string;
          description?: string;
          status: 'pending' | 'in_progress' | 'completed' | 'cancelled';
          priority: 'low' | 'medium' | 'high' | 'urgent';
          assigned_to?: string;
          created_by: string;
          project_id: string;
          due_date?: string;
          completed_at?: string;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          title: string;
          description?: string;
          status?: 'pending' | 'in_progress' | 'completed' | 'cancelled';
          priority?: 'low' | 'medium' | 'high' | 'urgent';
          assigned_to?: string;
          created_by: string;
          project_id: string;
          due_date?: string;
          completed_at?: string;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          title?: string;
          description?: string;
          status?: 'pending' | 'in_progress' | 'completed' | 'cancelled';
          priority?: 'low' | 'medium' | 'high' | 'urgent';
          assigned_to?: string;
          project_id?: string;
          due_date?: string;
          completed_at?: string;
          updated_at?: string;
        };
      };
    };
  };
}

// Utility functions for Supabase operations
export const checkSupabaseConnection = async () => {
  try {
    const { data, error } = await supabase.from('users').select('count').limit(1);
    return { connected: !error, error: error?.message };
  } catch (error) {
    return { connected: false, error: 'Failed to connect to Supabase' };
  }
};

export const getProjectsForUser = async (userId: string) => {
  const { data, error } = await supabase
    .from('projects')
    .select('*')
    .or(`created_by.eq.${userId},team_members.cs.{${userId}}`);

  return { data, error };
};

export const getChatMessages = async (userId: string, projectId?: string) => {
  let query = supabase
    .from('chat_messages')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: true });

  if (projectId) {
    query = query.eq('project_id', projectId);
  }

  const { data, error } = await query;
  return { data, error };
};

export const addChatMessage = async (message: Database['public']['Tables']['chat_messages']['Insert']) => {
  const { data, error } = await supabase
    .from('chat_messages')
    .insert(message)
    .select()
    .single();

  return { data, error };
};
