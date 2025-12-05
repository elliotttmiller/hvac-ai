import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Building2,
  Calendar,
  Users,
  FileText,
  AlertTriangle,
  CheckCircle2,
  Clock,
  MapPin,
  DollarSign,
  TrendingUp,
  Plus,
  Filter,
  Search,
  MoreHorizontal,
  Eye,
  Edit,
  Archive
} from "lucide-react";

interface Project {
  id: string;
  name: string;
  description: string;
  status: 'planning' | 'design' | 'construction' | 'completed';
  progress: number;
  startDate: Date;
  endDate: Date;
  budget: number;
  spent: number;
  location: string;
  teamMembers: number;
  documentsCount: number;
  phase: string;
  lastActivity: Date;
}

const mockProjects: Project[] = [
  {
    id: '1',
    name: 'Downtown Office Complex',
    description: 'Modern 25-story commercial office building with retail space',
    status: 'construction',
    progress: 85,
    startDate: new Date('2023-06-01'),
    endDate: new Date('2024-01-15'),
    budget: 45000000,
    spent: 38250000,
    location: 'Downtown District, City Center',
    teamMembers: 24,
    documentsCount: 156,
    phase: 'Foundation & Structure',
    lastActivity: new Date(Date.now() - 300000)
  },
  {
    id: '2',
    name: 'Residential Tower Alpha',
    description: 'Luxury residential tower with 200 units and amenities',
    status: 'design',
    progress: 45,
    startDate: new Date('2023-09-15'),
    endDate: new Date('2024-03-20'),
    budget: 28000000,
    spent: 12600000,
    location: 'Riverside District',
    teamMembers: 18,
    documentsCount: 89,
    phase: 'Design Development',
    lastActivity: new Date(Date.now() - 1800000)
  },
  {
    id: '3',
    name: 'Shopping Mall Renovation',
    description: 'Complete renovation and modernization of existing mall',
    status: 'planning',
    progress: 15,
    startDate: new Date('2024-01-01'),
    endDate: new Date('2024-06-30'),
    budget: 15000000,
    spent: 2250000,
    location: 'Suburban Plaza',
    teamMembers: 12,
    documentsCount: 34,
    phase: 'Planning',
    lastActivity: new Date(Date.now() - 3600000)
  },
  {
    id: '4',
    name: 'Industrial Warehouse',
    description: 'State-of-the-art logistics and distribution center',
    status: 'construction',
    progress: 72,
    startDate: new Date('2023-08-01'),
    endDate: new Date('2024-02-28'),
    budget: 18000000,
    spent: 12960000,
    location: 'Industrial Zone East',
    teamMembers: 16,
    documentsCount: 78,
    phase: 'MEP Installation',
    lastActivity: new Date(Date.now() - 900000)
  }
];

const getStatusBadge = (status: string) => {
  switch (status) {
    case 'planning':
      return <Badge variant="outline">Planning</Badge>;
    case 'design':
      return <Badge className="bg-blue-500">Design</Badge>;
    case 'construction':
      return <Badge className="bg-orange-500">Construction</Badge>;
    case 'completed':
      return <Badge className="bg-green-500">Completed</Badge>;
    default:
      return <Badge variant="secondary">Unknown</Badge>;
  }
};

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(amount);
};

export default function ProjectsPage() {
  const totalProjects = mockProjects.length;
  const activeProjects = mockProjects.filter(p => p.status !== 'completed').length;
  const totalBudget = mockProjects.reduce((sum, p) => sum + p.budget, 0);
  const totalSpent = mockProjects.reduce((sum, p) => sum + p.spent, 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Projects</h1>
          <p className="text-muted-foreground">
            Manage and monitor your construction projects
          </p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline">
            <Filter className="mr-2 h-4 w-4" />
            Filter
          </Button>
          <Button variant="outline">
            <Search className="mr-2 h-4 w-4" />
            Search
          </Button>
          <Button>
            <Plus className="mr-2 h-4 w-4" />
            New Project
          </Button>
        </div>
      </div>

      {/* Overview Stats */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Projects</CardTitle>
            <Building2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalProjects}</div>
            <p className="text-xs text-muted-foreground">
              {activeProjects} active projects
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Budget</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatCurrency(totalBudget)}</div>
            <p className="text-xs text-muted-foreground">
              Across all projects
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Spent</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatCurrency(totalSpent)}</div>
            <p className="text-xs text-muted-foreground">
              {((totalSpent / totalBudget) * 100).toFixed(1)}% of budget
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Progress</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {Math.round(mockProjects.reduce((sum, p) => sum + p.progress, 0) / mockProjects.length)}%
            </div>
            <p className="text-xs text-muted-foreground">
              Overall completion
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Projects Management */}
      <Tabs defaultValue="grid" className="space-y-4">
        <TabsList>
          <TabsTrigger value="grid">Grid View</TabsTrigger>
          <TabsTrigger value="list">List View</TabsTrigger>
          <TabsTrigger value="timeline">Timeline</TabsTrigger>
          <TabsTrigger value="kanban">Kanban</TabsTrigger>
        </TabsList>

        <TabsContent value="grid" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {mockProjects.map((project) => (
              <Card key={project.id} className="hover:shadow-md transition-shadow">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <CardTitle className="text-lg">{project.name}</CardTitle>
                      <CardDescription className="mt-1">
                        {project.description}
                      </CardDescription>
                    </div>
                    <Button variant="ghost" size="icon">
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </div>
                  <div className="flex items-center space-x-2 mt-2">
                    {getStatusBadge(project.status)}
                    <Badge variant="secondary">{project.phase}</Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Progress</span>
                      <span>{project.progress}%</span>
                    </div>
                    <Progress value={project.progress} className="h-2" />
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex items-center space-x-2">
                      <MapPin className="h-3 w-3 text-muted-foreground" />
                      <span className="text-muted-foreground truncate">
                        {project.location}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Users className="h-3 w-3 text-muted-foreground" />
                      <span className="text-muted-foreground">
                        {project.teamMembers} members
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Calendar className="h-3 w-3 text-muted-foreground" />
                      <span className="text-muted-foreground">
                        {project.endDate.toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <FileText className="h-3 w-3 text-muted-foreground" />
                      <span className="text-muted-foreground">
                        {project.documentsCount} docs
                      </span>
                    </div>
                  </div>

                  <div className="pt-2 border-t">
                    <div className="flex justify-between items-center text-sm">
                      <span className="text-muted-foreground">Budget</span>
                      <span className="font-medium">
                        {formatCurrency(project.spent)} / {formatCurrency(project.budget)}
                      </span>
                    </div>
                    <Progress
                      value={(project.spent / project.budget) * 100}
                      className="h-1 mt-1"
                    />
                  </div>

                  <div className="flex space-x-2">
                    <Button size="sm" variant="outline" className="flex-1">
                      <Eye className="mr-1 h-3 w-3" />
                      View
                    </Button>
                    <Button size="sm" variant="outline" className="flex-1">
                      <Edit className="mr-1 h-3 w-3" />
                      Edit
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="list" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Project List</CardTitle>
              <CardDescription>
                Detailed view of all construction projects
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockProjects.map((project) => (
                  <div key={project.id} className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50">
                    <div className="flex-1">
                      <div className="flex items-center space-x-4">
                        <div>
                          <h3 className="font-medium">{project.name}</h3>
                          <p className="text-sm text-muted-foreground">{project.location}</p>
                        </div>
                        <div className="flex space-x-2">
                          {getStatusBadge(project.status)}
                          <Badge variant="secondary">{project.phase}</Badge>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-8 text-sm">
                      <div className="text-center">
                        <p className="font-medium">{project.progress}%</p>
                        <p className="text-muted-foreground">Progress</p>
                      </div>
                      <div className="text-center">
                        <p className="font-medium">{formatCurrency(project.budget)}</p>
                        <p className="text-muted-foreground">Budget</p>
                      </div>
                      <div className="text-center">
                        <p className="font-medium">{project.teamMembers}</p>
                        <p className="text-muted-foreground">Team</p>
                      </div>
                      <div className="text-center">
                        <p className="font-medium">{project.endDate.toLocaleDateString()}</p>
                        <p className="text-muted-foreground">Due Date</p>
                      </div>
                      <div className="flex space-x-1">
                        <Button size="sm" variant="outline">
                          <Eye className="h-3 w-3" />
                        </Button>
                        <Button size="sm" variant="outline">
                          <Edit className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="timeline" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Project Timeline</CardTitle>
              <CardDescription>
                Visual timeline of project milestones and deadlines
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {mockProjects.map((project, index) => (
                  <div key={project.id} className="relative flex items-center space-x-4">
                    <div className="flex flex-col items-center">
                      <div className="w-4 h-4 bg-primary rounded-full"></div>
                      {index < mockProjects.length - 1 && (
                        <div className="w-0.5 h-16 bg-border mt-2"></div>
                      )}
                    </div>
                    <div className="flex-1 pb-8">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="font-medium">{project.name}</h3>
                          <p className="text-sm text-muted-foreground">{project.phase}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-medium">
                            {project.startDate.toLocaleDateString()} - {project.endDate.toLocaleDateString()}
                          </p>
                          <div className="flex items-center space-x-2 mt-1">
                            {getStatusBadge(project.status)}
                            <span className="text-sm text-muted-foreground">{project.progress}%</span>
                          </div>
                        </div>
                      </div>
                      <Progress value={project.progress} className="mt-2 h-2" />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="kanban" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-4">
            {['Planning', 'Design', 'Construction', 'Completed'].map((status) => (
              <Card key={status}>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">{status}</CardTitle>
                  <CardDescription>
                    {mockProjects.filter(p =>
                      p.status === status.toLowerCase().replace(' ', '')
                    ).length} projects
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {mockProjects
                    .filter(p => p.status === status.toLowerCase().replace(' ', ''))
                    .map((project) => (
                      <Card key={project.id} className="cursor-pointer hover:shadow-sm">
                        <CardContent className="p-3">
                          <h4 className="font-medium text-sm mb-2">{project.name}</h4>
                          <div className="space-y-2">
                            <div className="flex justify-between text-xs">
                              <span>Progress</span>
                              <span>{project.progress}%</span>
                            </div>
                            <Progress value={project.progress} className="h-1" />
                            <div className="flex items-center justify-between text-xs text-muted-foreground">
                              <span>{project.teamMembers} members</span>
                              <span>{formatCurrency(project.budget)}</span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
