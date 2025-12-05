'use client';

import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { RotateCcw, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';

export default function ThreeViewer() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);

    // Camera
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(5, 5, 5);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    mountRef.current.appendChild(renderer.domElement);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    // Sample building
    const buildingGroup = new THREE.Group();
    
    // Floor
    const floorGeometry = new THREE.BoxGeometry(10, 0.1, 10);
    const floorMaterial = new THREE.MeshStandardMaterial({ color: 0x808080 });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    buildingGroup.add(floor);

    // Walls
    const wallMaterial = new THREE.MeshStandardMaterial({ color: 0xcccccc });
    const wall1 = new THREE.Mesh(new THREE.BoxGeometry(10, 3, 0.2), wallMaterial);
    wall1.position.set(0, 1.5, -5);
    buildingGroup.add(wall1);

    // HVAC Unit
    const hvacGeometry = new THREE.BoxGeometry(1.5, 1, 1);
    const hvacMaterial = new THREE.MeshStandardMaterial({ color: 0x3b82f6 });
    const hvacUnit = new THREE.Mesh(hvacGeometry, hvacMaterial);
    hvacUnit.position.set(0, 3.5, 0);
    buildingGroup.add(hvacUnit);

    scene.add(buildingGroup);

    // Animation
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Cleanup
    return () => {
      mountRef.current?.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>3D BIM Viewer</CardTitle>
          <div className="flex gap-2">
            <Button variant="outline" size="sm"><RotateCcw className="h-4 w-4" /></Button>
            <Button variant="outline" size="sm"><ZoomIn className="h-4 w-4" /></Button>
            <Button variant="outline" size="sm"><ZoomOut className="h-4 w-4" /></Button>
            <Button variant="outline" size="sm"><Maximize2 className="h-4 w-4" /></Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div ref={mountRef} style={{ width: '100%', height: '500px' }} />
      </CardContent>
    </Card>
  );
}
