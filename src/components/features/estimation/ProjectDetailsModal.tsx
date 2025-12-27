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
      <DialogContent className="max-w-7xl w-[95vw] max-h-[90vh] overflow-hidden p-0 gap-0">
        {/* Hidden Title and Description for accessibility - needed by Dialog component */}
        <DialogTitle className="sr-only">{project.name}</DialogTitle>
        <DialogDescription className="sr-only">
          View and manage project details. You can delete this project below.
        </DialogDescription>

        <div className="flex flex-col h-full max-h-[90vh]">
          {/* Header */}
          <div className="flex items-center justify-between gap-4 p-4 sm:p-6 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="min-w-0 flex-1">
              <h3 className="text-lg sm:text-2xl font-semibold leading-tight truncate">{project.name}</h3>
              <div className="text-sm text-muted-foreground mt-1 truncate">{project.location ?? 'Unknown location'} • Zone {project.climateZone ?? '—'}</div>
            </div>

            <div className="flex items-center gap-2 flex-shrink-0">
              <Link href={`/workspace/${project.id}`}>
                <Button size="sm" className="hidden sm:inline-flex">Upload Blueprint</Button>
                <Button size="sm" className="sm:hidden">Upload</Button>
              </Link>
              <Button variant="outline" size="sm" onClick={() => {/* edit placeholder */}}>Edit</Button>
              <Button variant="ghost" size="sm" onClick={() => onOpenChange(false)}>Close</Button>
            </div>
          </div>

          {/* Body: responsive grid layout */}
          <div className="flex-1 overflow-auto">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 sm:gap-6 p-4 sm:p-6">
              {/* Left column: metadata - full width on mobile, sidebar on desktop */}
              <aside className="lg:col-span-1 space-y-4 sm:space-y-6 order-2 lg:order-1">
                <div className="space-y-2">
                  <div className="text-sm text-muted-foreground">Status</div>
                  <div className="flex items-center justify-between">
                    <div className="font-semibold">{project.status ?? 'Unknown'}</div>
                    <Badge variant={project.status === 'Completed' ? 'default' : project.status === 'In Progress' ? 'secondary' : 'outline'} className="ml-2">
                      {project.status ?? '—'}
                    </Badge>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="text-sm text-muted-foreground">Key Metrics</div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-muted/50 p-3 rounded-lg">
                      <div className="text-xs text-muted-foreground">Components</div>
                      <div className="font-semibold text-lg">{project.components ?? '—'}</div>
                    </div>
                    <div className="bg-muted/50 p-3 rounded-lg">
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
                  <div className="text-xs select-all font-mono bg-muted/50 p-2 rounded break-all">{project.id}</div>
                </div>
              </aside>

              {/* Right/main content - full width on mobile, 3 cols on desktop */}
              <main className="lg:col-span-3 space-y-4 sm:space-y-6 order-1 lg:order-2">
                {/* Overview / Description */}
                <section className="bg-muted/50 p-4 rounded-lg">
                  <h4 className="text-lg font-semibold">Overview</h4>
                  <p className="mt-2 text-sm text-muted-foreground">{project.description ?? 'No description provided for this project.'}</p>
                </section>

                {/* Analytics & Estimate breakdown */}
                <section className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                  <div className="bg-muted/50 p-4 rounded-lg min-h-[200px] flex flex-col">
                    <h5 className="font-medium">Estimate Breakdown</h5>
                    <div className="mt-3 text-sm text-muted-foreground flex-1">Line item estimates and a summary of costs will be shown here. Connect to estimate APIs to populate real data.</div>
                    <div className="mt-4 pt-4 border-t">
                      <div className="flex items-center justify-between">
                        <div className="text-xs text-muted-foreground">Subtotal</div>
                        <div className="font-semibold">${(project.estimatedCost ?? 0).toLocaleString()}</div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-muted/50 p-4 rounded-lg min-h-[200px]">
                    <h5 className="font-medium">Recent Activity</h5>
                    <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
                      <li>Blueprint uploaded — {friendlyDate(project.uploadedAt)}</li>
                      <li>Analysis run — {friendlyDate(project.analysisDate)}</li>
                      <li>Estimate updated — {friendlyDate(project.estimateUpdated)}</li>
                    </ul>
                  </div>
                </section>

                {/* Documents & actions */}
                <section className="bg-muted/50 p-4 rounded-lg">
                  <div className="flex items-center justify-between">
                    <h5 className="font-medium">Documents</h5>
                    <Link href={`/workspace/${project.id}`}>
                      <Button size="sm">Open Documents</Button>
                    </Link>
                  </div>
                  <div className="mt-3 grid gap-2">
                    {/* Placeholder documents list; replace with real data when available */}
                    <div className="flex items-center justify-between bg-background p-3 rounded border">
                      <div className="text-sm">HVAC_Blueprint.pdf</div>
                      <div className="text-xs text-muted-foreground">2.1 MB</div>
                    </div>
                  </div>
                </section>

                {/* Footer actions */}
                <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-end gap-2 pt-4 border-t">
                  <Link href={`/workspace/${project.id}`} className="sm:hidden">
                    <Button className="w-full">Upload Blueprint</Button>
                  </Link>
                  <div className="flex gap-2">
                    <Button variant="outline" className="flex-1 sm:flex-none">Export</Button>
                    <Button
                      variant="destructive"
                      className="flex-1 sm:flex-none"
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
                </div>
              </main>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
