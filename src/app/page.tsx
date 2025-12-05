import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Building2,
  Bot,
  FileText,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Users,
  TrendingUp,
  Activity,
  Zap,
  MessageSquare,
  Upload,
  Hammer,
} from "lucide-react";

export default function Dashboard() {
  return (
    <div className="space-y-6">
      {/* Welcome Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">
            Welcome to your AI-powered construction management platform
          </p>
        </div>
        <Button>
          <MessageSquare className="mr-2 h-4 w-4" />
          Chat with Suna AI
        </Button>
      </div>

      {/* Key Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Projects</CardTitle>
            <Building2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">
              +2 from last month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">AI Agents Active</CardTitle>
            <Bot className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">8/8</div>
            <p className="text-xs text-muted-foreground">
              All systems operational
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Documents Processed</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">1,234</div>
            <p className="text-xs text-muted-foreground">
              +180 today
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Compliance Score</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">98.5%</div>
            <p className="text-xs text-muted-foreground">
              Building codes compliance
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="agents">Agent Status</TabsTrigger>
          <TabsTrigger value="projects">Recent Projects</TabsTrigger>
          <TabsTrigger value="tasks">Task Queue</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
            {/* Project Activity */}
            <Card className="col-span-4">
              <CardHeader>
                <CardTitle>Recent Activity</CardTitle>
                <CardDescription>
                  Latest updates from your construction projects
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-4">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <div className="flex-1">
                    <p className="text-sm font-medium">Project Alpha BIM model updated</p>
                    <p className="text-xs text-muted-foreground">2 minutes ago</p>
                  </div>
                  <Badge variant="secondary">3D Model</Badge>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <div className="flex-1">
                    <p className="text-sm font-medium">CAD drawings processed by AI</p>
                    <p className="text-xs text-muted-foreground">5 minutes ago</p>
                  </div>
                  <Badge variant="secondary">OCR</Badge>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                  <div className="flex-1">
                    <p className="text-sm font-medium">Building code compliance check completed</p>
                    <p className="text-xs text-muted-foreground">10 minutes ago</p>
                  </div>
                  <Badge variant="secondary">Compliance</Badge>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                  <div className="flex-1">
                    <p className="text-sm font-medium">PM Bot assigned 12 new tasks</p>
                    <p className="text-xs text-muted-foreground">15 minutes ago</p>
                  </div>
                  <Badge variant="secondary">PM</Badge>
                </div>
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card className="col-span-3">
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
                <CardDescription>
                  Frequently used features and shortcuts
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button className="w-full justify-start" variant="outline">
                  <Upload className="mr-2 h-4 w-4" />
                  Upload Documents
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Bot className="mr-2 h-4 w-4" />
                  Chat with AI Agent
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Building2 className="mr-2 h-4 w-4" />
                  View 3D Models
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Hammer className="mr-2 h-4 w-4" />
                  Create New Project
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="agents" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Suna AI Assistant</CardTitle>
                <MessageSquare className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-sm">Online</span>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Processing 3 conversations
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Data Upload Bot</CardTitle>
                <Upload className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-sm">Online</span>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Processing 2 files
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">PM Bot</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-sm">Online</span>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Managing 8 projects
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Design Converter</CardTitle>
                <Zap className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                  <span className="text-sm">Busy</span>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Converting CAD to 3D
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="projects" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Active Projects</CardTitle>
              <CardDescription>
                Your current construction projects and their status
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h3 className="font-medium">Downtown Office Complex</h3>
                    <p className="text-sm text-muted-foreground">Phase: Foundation & Structure</p>
                  </div>
                  <div className="text-right">
                    <Badge className="mb-2">85% Complete</Badge>
                    <p className="text-xs text-muted-foreground">Due: Jan 15, 2024</p>
                  </div>
                </div>
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h3 className="font-medium">Residential Tower Alpha</h3>
                    <p className="text-sm text-muted-foreground">Phase: Design Development</p>
                  </div>
                  <div className="text-right">
                    <Badge variant="secondary" className="mb-2">45% Complete</Badge>
                    <p className="text-xs text-muted-foreground">Due: Mar 20, 2024</p>
                  </div>
                </div>
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h3 className="font-medium">Shopping Mall Renovation</h3>
                    <p className="text-sm text-muted-foreground">Phase: Planning</p>
                  </div>
                  <div className="text-right">
                    <Badge variant="outline" className="mb-2">15% Complete</Badge>
                    <p className="text-xs text-muted-foreground">Due: Jun 30, 2024</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tasks" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>AI Task Queue</CardTitle>
              <CardDescription>
                Current tasks being processed by AI agents
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center space-x-3 p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
                  <Activity className="h-4 w-4 text-blue-500" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">Processing architectural drawings</p>
                    <p className="text-xs text-muted-foreground">OCR extraction in progress...</p>
                  </div>
                  <Badge variant="secondary">Running</Badge>
                </div>
                <div className="flex items-center space-x-3 p-3 bg-yellow-50 dark:bg-yellow-950 rounded-lg">
                  <Clock className="h-4 w-4 text-yellow-500" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">3D model clash detection</p>
                    <p className="text-xs text-muted-foreground">Queued for processing...</p>
                  </div>
                  <Badge variant="outline">Pending</Badge>
                </div>
                <div className="flex items-center space-x-3 p-3 bg-green-50 dark:bg-green-950 rounded-lg">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">Building code compliance check</p>
                    <p className="text-xs text-muted-foreground">Completed successfully</p>
                  </div>
                  <Badge className="bg-green-500">Complete</Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
