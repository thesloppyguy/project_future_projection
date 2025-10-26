import { db } from './index';
import { users, organizations, rolePermissions } from './schema';
import { nanoid } from 'nanoid';
import bcrypt from 'bcryptjs';

async function seed() {
  console.log('🌱 Starting database seed...');

  try {
    // Create maintainer user
    const maintainerPassword = await bcrypt.hash('maintainer123!', 12);
    const [maintainer] = await db.insert(users).values({
      email: 'maintainer@platform.com',
      name: 'Platform Maintainer',
      password: maintainerPassword,
      emailVerified: true,
      status: 'active',
      role: 'maintainer',
    }).returning();

    console.log('✅ Created maintainer user:', maintainer.email);

    // Create default organization for testing
    const [defaultOrg] = await db.insert(organizations).values({
      name: 'Default Organization',
      slug: 'default-org',
      description: 'Default organization for testing',
      settings: {
        maxUsers: 100,
        maxTeams: 10,
        maxFileSize: 10 * 1024 * 1024, // 10MB
        allowedFileTypes: ['image/jpeg', 'image/png', 'application/pdf'],
        rateLimitPerHour: 1000,
        features: ['basic', 'advanced'],
      },
    }).returning();

    console.log('✅ Created default organization:', defaultOrg.name);

    // Seed role permissions
    const permissions = [
      // Maintainer permissions
      { role: 'maintainer', permission: 'platform:manage', resource: 'platform', action: 'manage' },
      { role: 'maintainer', permission: 'org:create', resource: 'organization', action: 'create' },
      { role: 'maintainer', permission: 'org:delete', resource: 'organization', action: 'delete' },
      { role: 'maintainer', permission: 'org:manage_all', resource: 'organization', action: 'manage' },
      { role: 'maintainer', permission: 'user:manage_all', resource: 'user', action: 'manage' },
      { role: 'maintainer', permission: 'user:impersonate', resource: 'user', action: 'impersonate' },
      { role: 'maintainer', permission: 'system:config', resource: 'system', action: 'manage' },
      { role: 'maintainer', permission: 'system:monitor', resource: 'system', action: 'read' },
      { role: 'maintainer', permission: 'maintainer:manage', resource: 'maintainer', action: 'manage' },

      // Org Admin permissions
      { role: 'org_admin', permission: 'org:manage', resource: 'organization', action: 'manage' },
      { role: 'org_admin', permission: 'org:view', resource: 'organization', action: 'read' },
      { role: 'org_admin', permission: 'team:create', resource: 'team', action: 'create' },
      { role: 'org_admin', permission: 'team:manage', resource: 'team', action: 'manage' },
      { role: 'org_admin', permission: 'team:view', resource: 'team', action: 'read' },
      { role: 'org_admin', permission: 'user:invite', resource: 'user', action: 'create' },
      { role: 'org_admin', permission: 'user:manage', resource: 'user', action: 'manage' },
      { role: 'org_admin', permission: 'role:assign', resource: 'role', action: 'manage' },
      { role: 'org_admin', permission: 'role:preview', resource: 'role', action: 'read' },
      { role: 'org_admin', permission: 'webhook:manage', resource: 'webhook', action: 'manage' },
      { role: 'org_admin', permission: 'file:upload', resource: 'file', action: 'create' },
      { role: 'org_admin', permission: 'logs:view', resource: 'logs', action: 'read' },

      // Team Admin permissions
      { role: 'team_admin', permission: 'team:manage', resource: 'team', action: 'manage', conditions: { teamOnly: true } },
      { role: 'team_admin', permission: 'team:view', resource: 'team', action: 'read', conditions: { teamOnly: true } },
      { role: 'team_admin', permission: 'user:invite', resource: 'user', action: 'create', conditions: { teamOnly: true } },
      { role: 'team_admin', permission: 'user:manage', resource: 'user', action: 'manage', conditions: { teamOnly: true } },
      { role: 'team_admin', permission: 'file:upload', resource: 'file', action: 'create' },
      { role: 'team_admin', permission: 'logs:view', resource: 'logs', action: 'read', conditions: { teamOnly: true } },

      // Team User permissions
      { role: 'team_user', permission: 'team:view', resource: 'team', action: 'read', conditions: { teamOnly: true } },
      { role: 'team_user', permission: 'user:view', resource: 'user', action: 'read', conditions: { ownOnly: true } },
      { role: 'team_user', permission: 'file:upload', resource: 'file', action: 'create' },
      { role: 'team_user', permission: 'logs:view', resource: 'logs', action: 'read', conditions: { ownOnly: true } },
    ];

    await db.insert(rolePermissions).values(permissions);

    console.log('✅ Seeded role permissions');

    console.log('🎉 Database seed completed successfully!');
    console.log('\n📋 Login credentials:');
    console.log('Email: maintainer@platform.com');
    console.log('Password: maintainer123!');
    console.log('\n⚠️  Please change the password after first login!');

  } catch (error) {
    console.error('❌ Seed failed:', error);
    throw error;
  }
}

// Run seed if called directly
if (require.main === module) {
  seed()
    .then(() => {
      console.log('Seed completed');
      process.exit(0);
    })
    .catch((error) => {
      console.error('Seed failed:', error);
      process.exit(1);
    });
}

export { seed };
