type Project = {
  id: string;
  name: string;
  location?: string;
  status?: string;
  components?: number;
  estimatedCost?: number;
  date?: string;
  climateZone?: string;
  description?: string;
};

// In-memory projects store for dev. This persists while the dev server process runs.
let projects: Project[] = [
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
];

export function getProjects() {
  return projects;
}

export function addProject(p: Omit<Project, 'id' | 'date'> & { date?: string }) {
  const project = {
    id: `${Date.now()}`,
    date: p.date ?? new Date().toISOString(),
    ...p,
  } as Project;
  projects = [project, ...projects];
  return project;
}

export function deleteProject(id: string) {
  const before = projects.length;
  projects = projects.filter((p) => p.id !== id);
  return projects.length < before;
}

