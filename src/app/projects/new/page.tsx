"use client";

import React from "react";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function NewProjectPage() {
  const [projectName, setProjectName] = useState("");
  const router = useRouter();
  const handleCreateProject = async () => {
    if (!projectName) return;
    try {
      const res = await fetch('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: projectName }),
      });
      if (!res.ok) throw new Error('Create failed');
      const data = await res.json();
      console.log('Created project', data.project);
      router.push('/projects');
    } catch (e) {
      console.error('Failed to create project', e);
      alert('Failed to create project');
    }
  };

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-4">Create New Project</h1>
      <div className="space-y-4">
        <Input
          placeholder="Enter project name"
          value={projectName}
          onChange={(e) => setProjectName(e.target.value)}
        />
        <Button onClick={handleCreateProject} disabled={!projectName}>
          Create Project
        </Button>
      </div>
    </div>
  );
}