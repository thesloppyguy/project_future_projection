'use client';

import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { api } from '@/lib/api';
import { Users, Building2, Users2, Mail, Activity, AlertCircle } from 'lucide-react';

export default function MaintainerDashboard() {
  const { data: overview, isLoading } = useQuery({
    queryKey: ['maintainer-overview'],
    queryFn: () => api.getMaintainerOverview(),
  });

  const { data: systemStatus } = useQuery({
    queryKey: ['system-status'],
    queryFn: () => api.getSystemStatus(),
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">Loading platform overview...</div>
      </div>
    );
  }

  const stats = overview?.stats || {};
  const recentActivity = overview?.recentActivity || [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Platform Overview</h1>
        <p className="text-muted-foreground">
          Monitor and manage the entire platform
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Users</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalUsers || 0}</div>
            <p className="text-xs text-muted-foreground">
              {stats.activeUsers || 0} active users
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Organizations</CardTitle>
            <Building2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalOrgs || 0}</div>
            <p className="text-xs text-muted-foreground">
              Active organizations
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Teams</CardTitle>
            <Users2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalTeams || 0}</div>
            <p className="text-xs text-muted-foreground">
              Across all organizations
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pending Invites</CardTitle>
            <Mail className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.pendingInvitations || 0}</div>
            <p className="text-xs text-muted-foreground">
              Awaiting acceptance
            </p>
          </CardContent>
        </Card>
      </div>

      {/* System Status */}
      {systemStatus && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              System Status
            </CardTitle>
            <CardDescription>
              Current system health and queue status
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <h4 className="font-medium">Queue Status</h4>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span>Pending:</span>
                    <Badge variant="outline">{systemStatus.queues?.pending || 0}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Active:</span>
                    <Badge variant="default">{systemStatus.queues?.active || 0}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Completed:</span>
                    <Badge variant="secondary">{systemStatus.queues?.completed || 0}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Failed:</span>
                    <Badge variant="destructive">{systemStatus.queues?.failed || 0}</Badge>
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <h4 className="font-medium">Webhooks</h4>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span>Total:</span>
                    <span>{systemStatus.webhooks?.total || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Active:</span>
                    <Badge variant="default">{systemStatus.webhooks?.active || 0}</Badge>
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <h4 className="font-medium">Scheduled Tasks</h4>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span>Total:</span>
                    <span>{systemStatus.scheduledTasks?.total || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Active:</span>
                    <Badge variant="default">{systemStatus.scheduledTasks?.active || 0}</Badge>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
          <CardDescription>
            Latest platform events and user actions
          </CardDescription>
        </CardHeader>
        <CardContent>
          {recentActivity.length > 0 ? (
            <div className="space-y-4">
              {recentActivity.map((activity: any) => (
                <div key={activity.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-muted rounded-full">
                      <Activity className="h-4 w-4" />
                    </div>
                    <div>
                      <p className="font-medium">{activity.action}</p>
                      <p className="text-sm text-muted-foreground">
                        {activity.resource} • {new Date(activity.createdAt).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <Badge variant="outline">
                    {activity.userId ? 'User Action' : 'System'}
                  </Badge>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              No recent activity
            </div>
          )}
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
          <CardDescription>
            Common administrative tasks
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Button variant="outline" className="h-auto p-4 flex flex-col items-center gap-2">
              <Building2 className="h-6 w-6" />
              <span>Create Organization</span>
            </Button>
            <Button variant="outline" className="h-auto p-4 flex flex-col items-center gap-2">
              <Users className="h-6 w-6" />
              <span>Manage Users</span>
            </Button>
            <Button variant="outline" className="h-auto p-4 flex flex-col items-center gap-2">
              <Mail className="h-6 w-6" />
              <span>Send Email</span>
            </Button>
            <Button variant="outline" className="h-auto p-4 flex flex-col items-center gap-2">
              <AlertCircle className="h-6 w-6" />
              <span>View Logs</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
