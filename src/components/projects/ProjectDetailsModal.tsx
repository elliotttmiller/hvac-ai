"use client";

import React from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";

interface Project {
  id: string;
  name: string;
  location?: string;
  status?: string;
  components?: number;
  estimatedCost?: number;
  date?: string;
  climateZone?: string;
  description?: string;
  uploadedAt?: string;
  analysisDate?: string;
  estimateUpdated?: string;
}


export default function ProjectDetailsModal({ project, open, onOpenChange, onDelete }: { project: Project | null; open: boolean; onOpenChange: (v: boolean) => void; onDelete?: (id: string) => void; }) {
  if (!project) return null;

  // Helper: friendly date
  const friendlyDate = (d?: string) => {
    try { return d ? new Date(d).toLocaleString() : '—'; } catch { return d ?? '—'; }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {/* Make the overlay + content cover the viewport; internal card is scrollable and centered */}
      <DialogContent className="inset-0 left-0 top-0 translate-x-0 translate-y-0 w-full h-full max-w-none sm:rounded-none p-0">
        <DialogTitle>{project.name}</DialogTitle>
        <DialogDescription>
          View and manage project details. You can delete this project below.
        </DialogDescription>
        <div className="mx-auto my-6 w-full max-w-7xl h-[calc(100vh-3rem)] overflow-hidden rounded-lg bg-popover shadow-lg ring-1 ring-border flex flex-col">

          {/* Header */}
          <div className="flex items-center justify-between gap-4 p-6 border-b">
            <div>
              <h3 className="text-2xl font-semibold leading-tight">{project.name}</h3>
              <div className="text-sm text-muted-foreground mt-1">{project.location ?? 'Unknown location'} • Zone {project.climateZone ?? '—'}</div>
            </div>

            <div className="flex items-center gap-2">
              <Link href={`/documents?projectId=${project.id}`}>
                <Button>Upload Blueprint</Button>
              </Link>
              <Button variant="outline" onClick={() => {/* edit placeholder */}}>Edit</Button>
              <Button variant="ghost" onClick={() => onOpenChange(false)}>Close</Button>
            </div>
          </div>

          {/* Body: left metadata column + right main content */}
          <div className="flex-1 overflow-auto">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 p-6">
              {/* Left column: metadata */}
              <aside className="md:col-span-1 space-y-6">
                <div className="space-y-2">
                  <div className="text-sm text-muted-foreground">Status</div>
                  <div className="flex items-center justify-between">
                    <div className="font-semibold">{project.status ?? 'Unknown'}</div>
                    <Badge variant={project.status === 'Completed' ? 'default' : project.status === 'In Progress' ? 'secondary' : 'outline'}>
                      {project.status ?? '—'}
                    </Badge>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="text-sm text-muted-foreground">Key Metrics</div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-background p-3 rounded">
                      <div className="text-xs text-muted-foreground">Components</div>
                      <div className="font-semibold text-lg">{project.components ?? '—'}</div>
                    </div>
                    <div className="bg-background p-3 rounded">
                      <div className="text-xs text-muted-foreground">Estimate</div>
                      <div className="font-semibold text-lg">${(project.estimatedCost ?? 0).toLocaleString()}</div>
                    </div>
                  </div>
                </div>

                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">Dates</div>
                  <div className="text-sm">Created: {friendlyDate(project.date)}</div>
                </div>

                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">Climate Zone</div>
                  <div className="font-medium">{project.climateZone ?? '—'}</div>
                </div>

                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">Project ID</div>
                  <div className="text-xs select-all font-mono">{project.id}</div>
                </div>
              </aside>

              {/* Right/main content */}
              <main className="md:col-span-3 space-y-6">
                {/* Overview / Description */}
                <section className="bg-background p-4 rounded shadow-sm">
                  <h4 className="text-lg font-semibold">Overview</h4>
                  <p className="mt-2 text-sm text-muted-foreground">{project.description ?? 'No description provided for this project.'}</p>
                </section>

                {/* Analytics & Estimate breakdown */}
                <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <div className="bg-background p-4 rounded shadow-sm h-48 flex flex-col">
                    <h5 className="font-medium">Estimate Breakdown</h5>
                    <div className="mt-3 text-sm text-muted-foreground">Line item estimates and a summary of costs will be shown here. Connect to estimate APIs to populate real data.</div>
                    <div className="mt-auto text-sm">
                      <div className="flex items-center justify-between">
                        <div className="text-xs text-muted-foreground">Subtotal</div>
                        <div className="font-semibold">${(project.estimatedCost ?? 0).toLocaleString()}</div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-background p-4 rounded shadow-sm h-48">
                    <h5 className="font-medium">Recent Activity</h5>
                    <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
                      <li>Blueprint uploaded — {friendlyDate(project.uploadedAt)}</li>
                      <li>Analysis run — {friendlyDate(project.analysisDate)}</li>
                      <li>Estimate updated — {friendlyDate(project.estimateUpdated)}</li>
                    </ul>
                  </div>
                </section>

                {/* Documents & actions */}
                <section className="bg-background p-4 rounded shadow-sm">
                  <div className="flex items-center justify-between">
                    <h5 className="font-medium">Documents</h5>
                    <Link href={`/documents?projectId=${project.id}`}>
                      <Button size="sm">Open Documents</Button>
                    </Link>
                  </div>
                  <div className="mt-3 grid gap-2">
                    {/* Placeholder documents list; replace with real data when available */}
                    <div className="flex items-center justify-between bg-muted p-3 rounded">
                      <div className="text-sm">HVAC_Blueprint.pdf</div>
                      <div className="text-xs text-muted-foreground">2.1 MB</div>
                    </div>
                  </div>
                </section>

                {/* Footer actions in main area */}
                <div className="flex items-center justify-end gap-2">
                  <Link href={`/documents?projectId=${project.id}`}>
                    <Button>Upload Blueprint</Button>
                  </Link>
                  <Button variant="outline">Export</Button>
                  <Button
                    variant="destructive"
                    onClick={() => {
                      if (!project.id) return;
                      if (!window.confirm('Are you sure you want to delete this project? This action cannot be undone.')) return;
                      if (onDelete) {
                        onDelete(project.id);
                      }
                      onOpenChange(false);
                    }}
                  >
                    Delete Project
                  </Button>
                </div>
              </main>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
