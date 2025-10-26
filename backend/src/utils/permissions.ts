import { db } from '../db';
import { rolePermissions, users, organizationMembers, teamMembers } from '../db/schema';
import { eq, and } from 'drizzle-orm';

export type UserRole = 'maintainer' | 'org_admin' | 'team_admin' | 'team_user';

export interface PermissionContext {
  userId: string;
  organizationId?: string;
  teamId?: string;
  resourceId?: string;
}

export class PermissionService {
  /**
   * Check if user has a specific permission
   */
  static async hasPermission(
    context: PermissionContext,
    permission: string,
    resource?: string,
    action?: string
  ): Promise<boolean> {
    const { userId, organizationId, teamId } = context;

    // Get user role
    const user = await db.select().from(users).where(eq(users.id, userId)).limit(1);
    if (!user.length) return false;

    const userRole = user[0].role as UserRole;

    // Maintainer has all permissions
    if (userRole === 'maintainer') {
      return true;
    }

    // Get role permissions
    const permissions = await db
      .select()
      .from(rolePermissions)
      .where(eq(rolePermissions.role, userRole));

    // Check if permission exists
    const hasPermission = permissions.some(p => {
      if (permission && p.permission !== permission) return false;
      if (resource && p.resource !== resource) return false;
      if (action && p.action !== action) return false;
      return true;
    });

    if (!hasPermission) return false;

    // Check conditions
    for (const perm of permissions) {
      if (perm.conditions) {
        const { ownOnly, teamOnly, orgOnly } = perm.conditions;

        if (ownOnly && context.resourceId !== userId) {
          continue;
        }

        if (teamOnly && teamId) {
          // Check if user is member of the team
          const teamMember = await db
            .select()
            .from(teamMembers)
            .where(and(
              eq(teamMembers.userId, userId),
              eq(teamMembers.teamId, teamId),
              eq(teamMembers.status, 'active')
            ))
            .limit(1);

          if (!teamMember.length) continue;
        }

        if (orgOnly && organizationId) {
          // Check if user is member of the organization
          const orgMember = await db
            .select()
            .from(organizationMembers)
            .where(and(
              eq(organizationMembers.userId, userId),
              eq(organizationMembers.organizationId, organizationId),
              eq(organizationMembers.status, 'active')
            ))
            .limit(1);

          if (!orgMember.length) continue;
        }
      }

      return true;
    }

    return false;
  }

  /**
   * Get all permissions for a user
   */
  static async getUserPermissions(context: PermissionContext): Promise<string[]> {
    const { userId } = context;

    const user = await db.select().from(users).where(eq(users.id, userId)).limit(1);
    if (!user.length) return [];

    const userRole = user[0].role as UserRole;

    // Maintainer has all permissions
    if (userRole === 'maintainer') {
      return ['*']; // Wildcard for all permissions
    }

    const permissions = await db
      .select()
      .from(rolePermissions)
      .where(eq(rolePermissions.role, userRole));

    return permissions.map(p => p.permission);
  }

  /**
   * Check if user is maintainer
   */
  static async isMaintainer(userId: string): Promise<boolean> {
    const user = await db.select().from(users).where(eq(users.id, userId)).limit(1);
    return user.length > 0 && user[0].role === 'maintainer';
  }

  /**
   * Check if user is org admin
   */
  static async isOrgAdmin(userId: string, organizationId: string): Promise<boolean> {
    const member = await db
      .select()
      .from(organizationMembers)
      .where(and(
        eq(organizationMembers.userId, userId),
        eq(organizationMembers.organizationId, organizationId),
        eq(organizationMembers.role, 'org_admin'),
        eq(organizationMembers.status, 'active')
      ))
      .limit(1);

    return member.length > 0;
  }

  /**
   * Check if user is team admin
   */
  static async isTeamAdmin(userId: string, teamId: string): Promise<boolean> {
    const member = await db
      .select()
      .from(teamMembers)
      .where(and(
        eq(teamMembers.userId, userId),
        eq(teamMembers.teamId, teamId),
        eq(teamMembers.role, 'team_admin'),
        eq(teamMembers.status, 'active')
      ))
      .limit(1);

    return member.length > 0;
  }
}
