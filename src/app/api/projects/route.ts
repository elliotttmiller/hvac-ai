import { NextResponse } from 'next/server';
import { getProjects, addProject, deleteProject } from './store';

export async function GET() {
  const projects = getProjects();
  return NextResponse.json({ projects });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const created = addProject({ name: body.name, location: body.location, status: body.status, components: body.components, estimatedCost: body.estimatedCost, climateZone: body.climateZone, description: body.description });
    return NextResponse.json({ project: created }, { status: 201 });
  } catch (e) {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const projectId = searchParams.get('id');
  if (!projectId) {
    return NextResponse.json({ error: 'Missing project id' }, { status: 400 });
  }
  const ok = deleteProject(projectId);
  if (!ok) {
    return NextResponse.json({ error: 'Project not found' }, { status: 404 });
  }
  return NextResponse.json({ success: true });
}

