"use client";

import React from "react";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";

export default function NewProjectPage() {
  const [projectName, setProjectName] = useState("");
  const router = useRouter();
  const handleCreateProject = async () => {
    if (!projectName) return;
    try {
      const res = await fetch('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '69420' },
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
    <div className="container mx-auto py-8 space-y-6">
      <div className="flex items-center gap-4">
        <Link href="/projects">
          <Button variant="outline" size="sm">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Projects
          </Button>
        </Link>
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Create New Project</h1>
          <p className="text-muted-foreground">Set up a new HVAC blueprint analysis project</p>
        </div>
      </div>
      <div className="max-w-md space-y-4">
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

