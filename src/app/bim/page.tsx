'use client';

import React from 'react';
import ThreeViewer from '@/components/bim/ThreeViewer';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Wind, Layers, Box } from 'lucide-react';

export default function BIMPage() {
  return (
    <div className="container mx-auto py-8 space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-2">3D BIM Viewer</h1>
        <p className="text-muted-foreground">
          Visualize HVAC systems in 3D with interactive building models
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card>
          <CardHeader className="pb-3">
            <Wind className="h-8 w-8 text-blue-500 mb-2" />
            <CardTitle className="text-sm">HVAC Components</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">12</p>
            <p className="text-xs text-muted-foreground">Detected in model</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <Layers className="h-8 w-8 text-green-500 mb-2" />
            <CardTitle className="text-sm">Layers</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">8</p>
            <p className="text-xs text-muted-foreground">Active layers</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <Box className="h-8 w-8 text-purple-500 mb-2" />
            <CardTitle className="text-sm">Elements</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">156</p>
            <p className="text-xs text-muted-foreground">Total building elements</p>
          </CardContent>
        </Card>
      </div>

      <ThreeViewer />
    </div>
  );
}
