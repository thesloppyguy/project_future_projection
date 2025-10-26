import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Building2, Users, Shield, Mail } from 'lucide-react';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Multi-tenant Auth Platform
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            A comprehensive platform with role-based access control, multi-tenancy, and advanced features
          </p>
          <div className="flex gap-4 justify-center">
            <Link href="/login">
              <Button size="lg">Sign In</Button>
            </Link>
            <Link href="/maintainer">
              <Button variant="outline" size="lg">Platform Admin</Button>
            </Link>
          </div>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          <Card>
            <CardHeader>
              <Building2 className="h-8 w-8 text-blue-600 mb-2" />
              <CardTitle>Multi-tenant</CardTitle>
              <CardDescription>
                Organizations with isolated data and settings
              </CardDescription>
            </CardHeader>
          </Card>

          <Card>
            <CardHeader>
              <Users className="h-8 w-8 text-green-600 mb-2" />
              <CardTitle>Role-based Access</CardTitle>
              <CardDescription>
                Granular permissions with maintainer, org admin, team admin, and user roles
              </CardDescription>
            </CardHeader>
          </Card>

          <Card>
            <CardHeader>
              <Shield className="h-8 w-8 text-purple-600 mb-2" />
              <CardTitle>Security</CardTitle>
              <CardDescription>
                Audit logs, rate limiting, and secure authentication
              </CardDescription>
            </CardHeader>
          </Card>

          <Card>
            <CardHeader>
              <Mail className="h-8 w-8 text-orange-600 mb-2" />
              <CardTitle>Notifications</CardTitle>
              <CardDescription>
                Email integration, webhooks, and scheduled tasks
              </CardDescription>
            </CardHeader>
          </Card>
        </div>

        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Features
          </h2>
          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <div className="text-left">
              <h3 className="font-semibold mb-2">Authentication & Authorization</h3>
              <ul className="text-gray-600 space-y-1">
                <li>• Better Auth integration</li>
                <li>• Invite-only registration</li>
                <li>• Password reset flow</li>
                <li>• Role-based permissions</li>
                <li>• User impersonation (maintainers)</li>
              </ul>
            </div>
            <div className="text-left">
              <h3 className="font-semibold mb-2">Platform Management</h3>
              <ul className="text-gray-600 space-y-1">
                <li>• Organization management</li>
                <li>• Team management</li>
                <li>• Member invitations</li>
                <li>• File uploads</li>
                <li>• Webhook system</li>
              </ul>
            </div>
            <div className="text-left">
              <h3 className="font-semibold mb-2">Monitoring & Logs</h3>
              <ul className="text-gray-600 space-y-1">
                <li>• Audit logging</li>
                <li>• Data sync logs</li>
                <li>• Training logs</li>
                <li>• Notification logs</li>
                <li>• System monitoring</li>
              </ul>
            </div>
            <div className="text-left">
              <h3 className="font-semibold mb-2">Advanced Features</h3>
              <ul className="text-gray-600 space-y-1">
                <li>• Scheduled tasks</li>
                <li>• Queue management</li>
                <li>• Rate limiting</li>
                <li>• Custom role groups</li>
                <li>• Role preview mode</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}