'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { 
  Plus,
  Folder,
  Grid3x3,
} from 'lucide-react';
import { ProjectCard } from '@/components/features/dashboard/ProjectCard';

interface Project {
  id: string;
  name: string;
  location?: string;
  status?: string;
  createdAt?: string;
  progress?: number;
  teamMembers?: Array<{
    name: string;
    avatar?: string;
  }>;
  components?: number;
  documentsCount?: number;
}

/**
 * DashboardContent - The "Home Base" view
 * Displays all projects in a Snetch-style card grid
 */
export default function DashboardContent() {
  const router = useRouter();
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch projects from API
  useEffect(() => {
    let mounted = true;
    setLoading(true);
    
    fetch('/api/projects', { 
      headers: { 'ngrok-skip-browser-warning': '69420' } 
    })
      .then((res) => res.json())
      .then((data) => {
        if (!mounted) return;
        const projectsData = data.projects || [];
        
        // Enhance projects with additional data
        const enhancedProjects = projectsData.map((proj: Project) => ({
          ...proj,
          progress: Math.floor(Math.random() * 100), // Mock progress
          components: Math.floor(Math.random() * 50), // Mock components count
          documentsCount: Math.floor(Math.random() * 10), // Mock documents count
          teamMembers: [], // No team members for now
        }));
        
        setProjects(enhancedProjects);
      })
      .catch((err) => {
        console.error('Failed to load projects', err);
        setProjects([]);
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    
    return () => {
      mounted = false;
    };
  }, []);

  const handleCreateProject = () => {
    router.push('/projects/new');
  };

  if (loading) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-slate-950">
        <div className="text-center space-y-3">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="text-slate-400 text-sm">Loading projects...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full w-full overflow-auto bg-slate-950">
      <div className="max-w-7xl mx-auto p-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <Folder className="w-8 h-8 text-blue-400" />
              <h1 className="text-3xl font-bold text-white">Projects</h1>
            </div>
            <Button
              onClick={handleCreateProject}
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              <Plus className="w-4 h-4 mr-2" />
              Add New
            </Button>
          </div>
          <p className="text-slate-400 text-sm">
            {projects.length} {projects.length === 1 ? 'project' : 'projects'} • Click a card to open workspace
          </p>
        </motion.div>

        {/* Projects Grid */}
        {projects.length > 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {projects.map((project, index) => (
              <motion.div
                key={project.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <ProjectCard
                  id={project.id}
                  name={project.name}
                  location={project.location}
                  status={project.status}
                  progress={project.progress}
                  createdAt={project.createdAt}
                  teamMembers={project.teamMembers}
                  components={project.components}
                  documentsCount={project.documentsCount}
                />
              </motion.div>
            ))}
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center py-20"
          >
            <Grid3x3 className="w-20 h-20 mx-auto mb-4 text-slate-700" />
            <h3 className="text-xl font-semibold text-white mb-2">No projects yet</h3>
            <p className="text-slate-400 mb-6">Create your first project to get started</p>
            <Button
              onClick={handleCreateProject}
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              <Plus className="w-4 h-4 mr-2" />
              Create Project
            </Button>
          </motion.div>
        )}
      </div>
    </div>
  );
}

