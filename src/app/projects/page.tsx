'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Wind, Building2, DollarSign, Calendar, MapPin, FileText, Plus, Upload } from 'lucide-react';
import Link from 'next/link';

const projects = [
  {
    id: '1',
    name: 'Downtown Office Building',
    location: 'Chicago, IL',
    status: 'In Progress',
    components: 42,
    estimatedCost: 185000,
    date: '2024-12-01',
    climateZone: '5A',
  },
  {
    id: '2',
    name: 'Warehouse Facility',
    location: 'Phoenix, AZ',
    status: 'Completed',
    components: 28,
    estimatedCost: 125000,
    date: '2024-11-28',
    climateZone: '2B',
  },
  {
    id: '3',
    name: 'Retail Complex',
    location: 'Miami, FL',
    status: 'Analysis',
    components: 65,
    estimatedCost: 310000,
    date: '2024-12-03',
    climateZone: '1A',
  },
];

export default function ProjectsPage() {
  return (
    <div className="container mx-auto py-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">HVAC Projects</h1>
          <p className="text-muted-foreground">Manage your HVAC blueprint analyses and estimates</p>
        </div>
        <Link href="/documents">
          <Button>
            <Plus className="mr-2 h-4 w-4" />
            New Project
          </Button>
        </Link>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {projects.map((project) => (
          <Card key={project.id} className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <CardTitle className="text-lg mb-1">{project.name}</CardTitle>
                  <CardDescription className="flex items-center gap-1">
                    <MapPin className="h-3 w-3" />
                    {project.location}
                  </CardDescription>
                </div>
                <Badge variant={project.status === 'Completed' ? 'default' : project.status === 'In Progress' ? 'secondary' : 'outline'}>
                  {project.status}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Wind className="h-3 w-3" />
                    Components
                  </div>
                  <div className="text-2xl font-bold">{project.components}</div>
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <DollarSign className="h-3 w-3" />
                    Estimate
                  </div>
                  <div className="text-2xl font-bold">${(project.estimatedCost / 1000).toFixed(0)}K</div>
                </div>
              </div>

              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-1 text-muted-foreground">
                  <Calendar className="h-3 w-3" />
                  {new Date(project.date).toLocaleDateString()}
                </div>
                <Badge variant="outline" className="text-xs">Zone {project.climateZone}</Badge>
              </div>

              <Button variant="outline" className="w-full" size="sm">
                <FileText className="mr-2 h-3 w-3" />
                View Details
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>

      {projects.length === 0 && (
        <Card className="p-12">
          <div className="text-center space-y-4">
            <Building2 className="h-12 w-12 mx-auto text-muted-foreground" />
            <div>
              <h3 className="text-lg font-semibold">No projects yet</h3>
              <p className="text-sm text-muted-foreground">Upload your first HVAC blueprint to get started</p>
            </div>
            <Link href="/documents">
              <Button>
                <Upload className="mr-2 h-4 w-4" />
                Upload Blueprint
              </Button>
            </Link>
          </div>
        </Card>
      )}
    </div>
  );
}
