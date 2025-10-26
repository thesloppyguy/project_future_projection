'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { api } from '@/lib/api';
import { Plus, Building2, Users, Calendar } from 'lucide-react';

export default function OrganizationsPage() {
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [newOrg, setNewOrg] = useState({
    name: '',
    slug: '',
    description: '',
  });

  const queryClient = useQueryClient();

  const { data: organizations, isLoading } = useQuery({
    queryKey: ['organizations'],
    queryFn: () => api.getOrganizations(),
  });

  const createOrgMutation = useMutation({
    mutationFn: (data: { name: string; slug: string; description?: string }) =>
      api.createOrganization(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['organizations'] });
      setIsCreateDialogOpen(false);
      setNewOrg({ name: '', slug: '', description: '' });
    },
  });

  const handleCreateOrg = (e: React.FormEvent) => {
    e.preventDefault();
    if (newOrg.name && newOrg.slug) {
      createOrgMutation.mutate(newOrg);
    }
  };

  const generateSlug = (name: string) => {
    return name
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .trim();
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Organizations</h1>
          <p className="text-muted-foreground">
            Manage all organizations on the platform
          </p>
        </div>
        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Create Organization
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Organization</DialogTitle>
              <DialogDescription>
                Create a new organization for your platform
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleCreateOrg} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">Organization Name</Label>
                <Input
                  id="name"
                  value={newOrg.name}
                  onChange={(e) => {
                    setNewOrg({
                      ...newOrg,
                      name: e.target.value,
                      slug: generateSlug(e.target.value),
                    });
                  }}
                  placeholder="Acme Corporation"
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="slug">Slug</Label>
                <Input
                  id="slug"
                  value={newOrg.slug}
                  onChange={(e) => setNewOrg({ ...newOrg, slug: e.target.value })}
                  placeholder="acme-corporation"
                  required
                />
                <p className="text-xs text-muted-foreground">
                  Used in URLs and must be unique
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="description">Description (Optional)</Label>
                <Input
                  id="description"
                  value={newOrg.description}
                  onChange={(e) => setNewOrg({ ...newOrg, description: e.target.value })}
                  placeholder="Brief description of the organization"
                />
              </div>
              <div className="flex justify-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setIsCreateDialogOpen(false)}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={createOrgMutation.isPending}>
                  {createOrgMutation.isPending ? 'Creating...' : 'Create Organization'}
                </Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {isLoading ? (
        <div className="text-center py-8">Loading organizations...</div>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>All Organizations</CardTitle>
            <CardDescription>
              {organizations?.organizations?.length || 0} organizations found
            </CardDescription>
          </CardHeader>
          <CardContent>
            {organizations?.organizations?.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Organization</TableHead>
                    <TableHead>Members</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {organizations.organizations.map((org: any) => (
                    <TableRow key={org.id}>
                      <TableCell>
                        <div className="flex items-center gap-3">
                          <div className="p-2 bg-muted rounded-full">
                            <Building2 className="h-4 w-4" />
                          </div>
                          <div>
                            <p className="font-medium">{org.name}</p>
                            <p className="text-sm text-muted-foreground">
                              {org.slug}
                            </p>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1">
                          <Users className="h-4 w-4 text-muted-foreground" />
                          <span>{org.userCount}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant={org.status === 'active' ? 'default' : 'secondary'}>
                          {org.status}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1 text-sm text-muted-foreground">
                          <Calendar className="h-4 w-4" />
                          {new Date(org.createdAt).toLocaleDateString()}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex gap-2">
                          <Button variant="outline" size="sm">
                            View
                          </Button>
                          <Button variant="outline" size="sm">
                            Edit
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                No organizations found. Create your first organization to get started.
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
