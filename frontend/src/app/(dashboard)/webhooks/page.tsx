'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Plus, TestTube, Trash2, Edit, Eye, RotateCcw, BarChart3 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { api } from '@/lib/api';

interface Webhook {
  id: string;
  name: string;
  url: string;
  events: string[];
  isActive: boolean;
  retryCount: string;
  timeout: string;
  createdAt: string;
  updatedAt: string;
}

interface WebhookEvent {
  id: string;
  event: string;
  status: 'pending' | 'delivered' | 'failed';
  responseCode?: string;
  attempts: string;
  deliveredAt?: string;
  error?: string;
  createdAt: string;
}

interface WebhookStats {
  totalEvents: number;
  successfulDeliveries: number;
  failedDeliveries: number;
  successRate: number;
  averageResponseTime: number;
  lastDelivery: string | null;
  eventsByStatus: Record<string, number>;
  eventsByDay: Array<{ date: string; count: number; success: number; failed: number }>;
}

export default function WebhooksPage() {
  const [webhooks, setWebhooks] = useState<Webhook[]>([]);
  const [selectedWebhook, setSelectedWebhook] = useState<Webhook | null>(null);
  const [webhookEvents, setWebhookEvents] = useState<WebhookEvent[]>([]);
  const [webhookStats, setWebhookStats] = useState<WebhookStats | null>(null);
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isTestDialogOpen, setIsTestDialogOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [testLoading, setTestLoading] = useState(false);
  const { toast } = useToast();

  // Form states
  const [webhookForm, setWebhookForm] = useState({
    name: '',
    url: '',
    events: [] as string[],
    secret: '',
    isActive: true,
    retryCount: 3,
    timeout: 30,
  });

  const [testForm, setTestForm] = useState({
    eventType: 'user',
    eventName: 'user.created',
    payload: '{}',
  });

  const availableEvents = [
    { category: 'User Events', events: ['user.created', 'user.updated', 'user.deleted', 'user.activated'] },
    { category: 'Order Events', events: ['order.created', 'order.updated', 'order.cancelled', 'order.completed'] },
    { category: 'Payment Events', events: ['payment.created', 'payment.completed', 'payment.failed', 'payment.refunded'] },
    { category: 'System Events', events: ['system.maintenance', 'system.error', 'system.alert'] },
  ];

  useEffect(() => {
    fetchWebhooks();
  }, []);

  const fetchWebhooks = async () => {
    try {
      const response = await api.get('/webhooks');
      setWebhooks(response.data.webhooks);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to fetch webhooks',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const fetchWebhookEvents = async (webhookId: string) => {
    try {
      const response = await api.get(`/webhooks/${webhookId}/events`);
      setWebhookEvents(response.data.events);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to fetch webhook events',
        variant: 'destructive',
      });
    }
  };

  const fetchWebhookStats = async (webhookId: string) => {
    try {
      const response = await api.get(`/webhooks/${webhookId}/stats`);
      setWebhookStats(response.data.stats);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to fetch webhook statistics',
        variant: 'destructive',
      });
    }
  };

  const createWebhook = async () => {
    try {
      await api.post('/webhooks', webhookForm);
      toast({
        title: 'Success',
        description: 'Webhook created successfully',
      });
      setIsCreateDialogOpen(false);
      setWebhookForm({
        name: '',
        url: '',
        events: [],
        secret: '',
        isActive: true,
        retryCount: 3,
        timeout: 30,
      });
      fetchWebhooks();
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to create webhook',
        variant: 'destructive',
      });
    }
  };

  const updateWebhook = async (webhookId: string, data: Partial<Webhook>) => {
    try {
      await api.put(`/webhooks/${webhookId}`, data);
      toast({
        title: 'Success',
        description: 'Webhook updated successfully',
      });
      fetchWebhooks();
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to update webhook',
        variant: 'destructive',
      });
    }
  };

  const deleteWebhook = async (webhookId: string) => {
    try {
      await api.delete(`/webhooks/${webhookId}`);
      toast({
        title: 'Success',
        description: 'Webhook deleted successfully',
      });
      fetchWebhooks();
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to delete webhook',
        variant: 'destructive',
      });
    }
  };

  const testWebhook = async () => {
    if (!selectedWebhook) return;

    setTestLoading(true);
    try {
      const payload = JSON.parse(testForm.payload);
      const response = await api.post(`/webhooks/${selectedWebhook.id}/test`, {
        eventType: testForm.eventType,
        eventName: testForm.eventName,
        payload,
      });

      toast({
        title: 'Test Completed',
        description: response.data.result.success 
          ? 'Webhook test successful' 
          : `Webhook test failed: ${response.data.result.error}`,
        variant: response.data.result.success ? 'default' : 'destructive',
      });

      setIsTestDialogOpen(false);
      fetchWebhookEvents(selectedWebhook.id);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to test webhook',
        variant: 'destructive',
      });
    } finally {
      setTestLoading(false);
    }
  };

  const retryWebhookEvent = async (webhookId: string, eventId: string) => {
    try {
      await api.post(`/webhooks/${webhookId}/events/${eventId}/retry`);
      toast({
        title: 'Success',
        description: 'Webhook retry initiated',
      });
      fetchWebhookEvents(webhookId);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to retry webhook',
        variant: 'destructive',
      });
    }
  };

  const getStatusBadge = (status: string) => {
    const variants = {
      delivered: 'default',
      failed: 'destructive',
      pending: 'secondary',
    } as const;

    return (
      <Badge variant={variants[status as keyof typeof variants] || 'secondary'}>
        {status}
      </Badge>
    );
  };

  if (loading) {
    return <div className="p-6">Loading webhooks...</div>;
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Webhooks</h1>
          <p className="text-muted-foreground">
            Manage webhook endpoints and monitor delivery status
          </p>
        </div>
        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="w-4 h-4 mr-2" />
              Create Webhook
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Create New Webhook</DialogTitle>
              <DialogDescription>
                Configure a new webhook endpoint to receive events
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div>
                <Label htmlFor="name">Name</Label>
                <Input
                  id="name"
                  value={webhookForm.name}
                  onChange={(e) => setWebhookForm({ ...webhookForm, name: e.target.value })}
                  placeholder="My Webhook"
                />
              </div>
              <div>
                <Label htmlFor="url">URL</Label>
                <Input
                  id="url"
                  value={webhookForm.url}
                  onChange={(e) => setWebhookForm({ ...webhookForm, url: e.target.value })}
                  placeholder="https://example.com/webhook"
                />
              </div>
              <div>
                <Label>Events</Label>
                <div className="space-y-2">
                  {availableEvents.map((category) => (
                    <div key={category.category}>
                      <h4 className="text-sm font-medium">{category.category}</h4>
                      <div className="flex flex-wrap gap-2">
                        {category.events.map((event) => (
                          <Button
                            key={event}
                            variant={webhookForm.events.includes(event) ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => {
                              const events = webhookForm.events.includes(event)
                                ? webhookForm.events.filter(e => e !== event)
                                : [...webhookForm.events, event];
                              setWebhookForm({ ...webhookForm, events });
                            }}
                          >
                            {event}
                          </Button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <Label htmlFor="secret">Secret (Optional)</Label>
                <Input
                  id="secret"
                  type="password"
                  value={webhookForm.secret}
                  onChange={(e) => setWebhookForm({ ...webhookForm, secret: e.target.value })}
                  placeholder="Webhook secret for signature validation"
                />
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="isActive"
                  checked={webhookForm.isActive}
                  onCheckedChange={(checked) => setWebhookForm({ ...webhookForm, isActive: checked })}
                />
                <Label htmlFor="isActive">Active</Label>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="retryCount">Retry Count</Label>
                  <Input
                    id="retryCount"
                    type="number"
                    min="0"
                    max="10"
                    value={webhookForm.retryCount}
                    onChange={(e) => setWebhookForm({ ...webhookForm, retryCount: parseInt(e.target.value) })}
                  />
                </div>
                <div>
                  <Label htmlFor="timeout">Timeout (seconds)</Label>
                  <Input
                    id="timeout"
                    type="number"
                    min="1"
                    max="300"
                    value={webhookForm.timeout}
                    onChange={(e) => setWebhookForm({ ...webhookForm, timeout: parseInt(e.target.value) })}
                  />
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={createWebhook}>Create Webhook</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <Tabs defaultValue="webhooks" className="space-y-4">
        <TabsList>
          <TabsTrigger value="webhooks">Webhooks</TabsTrigger>
          <TabsTrigger value="events" disabled={!selectedWebhook}>
            Events
          </TabsTrigger>
          <TabsTrigger value="stats" disabled={!selectedWebhook}>
            Statistics
          </TabsTrigger>
        </TabsList>

        <TabsContent value="webhooks" className="space-y-4">
          <div className="grid gap-4">
            {webhooks.map((webhook) => (
              <Card key={webhook.id}>
                <CardHeader>
                  <div className="flex justify-between items-start">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        {webhook.name}
                        {getStatusBadge(webhook.isActive ? 'delivered' : 'pending')}
                      </CardTitle>
                      <CardDescription>{webhook.url}</CardDescription>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          setSelectedWebhook(webhook);
                          fetchWebhookEvents(webhook.id);
                        }}
                      >
                        <Eye className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          setSelectedWebhook(webhook);
                          setIsTestDialogOpen(true);
                        }}
                      >
                        <TestTube className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => updateWebhook(webhook.id, { isActive: !webhook.isActive })}
                      >
                        <Edit className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => deleteWebhook(webhook.id)}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex gap-4 text-sm text-muted-foreground">
                      <span>Events: {webhook.events.length}</span>
                      <span>Retries: {webhook.retryCount}</span>
                      <span>Timeout: {webhook.timeout}s</span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {webhook.events.map((event) => (
                        <Badge key={event} variant="outline" className="text-xs">
                          {event}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="events" className="space-y-4">
          {selectedWebhook && (
            <>
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold">Events for {selectedWebhook.name}</h3>
                <Button onClick={() => fetchWebhookEvents(selectedWebhook.id)}>
                  Refresh
                </Button>
              </div>
              <Card>
                <CardContent className="p-0">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Event</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Response Code</TableHead>
                        <TableHead>Attempts</TableHead>
                        <TableHead>Delivered At</TableHead>
                        <TableHead>Error</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {webhookEvents.map((event) => (
                        <TableRow key={event.id}>
                          <TableCell className="font-medium">{event.event}</TableCell>
                          <TableCell>{getStatusBadge(event.status)}</TableCell>
                          <TableCell>{event.responseCode || '-'}</TableCell>
                          <TableCell>{event.attempts}</TableCell>
                          <TableCell>
                            {event.deliveredAt 
                              ? new Date(event.deliveredAt).toLocaleString()
                              : '-'
                            }
                          </TableCell>
                          <TableCell className="max-w-xs truncate">
                            {event.error || '-'}
                          </TableCell>
                          <TableCell>
                            {event.status === 'failed' && (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => retryWebhookEvent(selectedWebhook.id, event.id)}
                              >
                                <RotateCcw className="w-4 h-4" />
                              </Button>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </>
          )}
        </TabsContent>

        <TabsContent value="stats" className="space-y-4">
          {selectedWebhook && webhookStats && (
            <>
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold">Statistics for {selectedWebhook.name}</h3>
                <Button onClick={() => fetchWebhookStats(selectedWebhook.id)}>
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Refresh
                </Button>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Total Events</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{webhookStats.totalEvents}</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{webhookStats.successRate.toFixed(1)}%</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Avg Response Time</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{webhookStats.averageResponseTime.toFixed(0)}ms</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Last Delivery</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-sm">
                      {webhookStats.lastDelivery 
                        ? new Date(webhookStats.lastDelivery).toLocaleString()
                        : 'Never'
                      }
                    </div>
                  </CardContent>
                </Card>
              </div>
            </>
          )}
        </TabsContent>
      </Tabs>

      {/* Test Webhook Dialog */}
      <Dialog open={isTestDialogOpen} onOpenChange={setIsTestDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Test Webhook</DialogTitle>
            <DialogDescription>
              Send a test event to {selectedWebhook?.name}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label htmlFor="eventType">Event Type</Label>
              <Select
                value={testForm.eventType}
                onValueChange={(value) => setTestForm({ ...testForm, eventType: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="user">User</SelectItem>
                  <SelectItem value="order">Order</SelectItem>
                  <SelectItem value="payment">Payment</SelectItem>
                  <SelectItem value="system">System</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="eventName">Event Name</Label>
              <Input
                id="eventName"
                value={testForm.eventName}
                onChange={(e) => setTestForm({ ...testForm, eventName: e.target.value })}
                placeholder="user.created"
              />
            </div>
            <div>
              <Label htmlFor="payload">Payload (JSON)</Label>
              <Textarea
                id="payload"
                value={testForm.payload}
                onChange={(e) => setTestForm({ ...testForm, payload: e.target.value })}
                placeholder='{"userId": "123", "email": "test@example.com"}'
                rows={4}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsTestDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={testWebhook} disabled={testLoading}>
              {testLoading ? 'Testing...' : 'Test Webhook'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
