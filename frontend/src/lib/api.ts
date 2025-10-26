import { authClient } from './auth';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000';

class ApiClient {
  private baseURL: string;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const session = await authClient.getSession();
    
    const url = `${this.baseURL}${endpoint}`;
    const config: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(session?.token && { Authorization: `Bearer ${session.token}` }),
        ...options.headers,
      },
    };

    const response = await fetch(url, config);

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: 'An error occurred' }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // Auth methods
  async getMe() {
    return this.request('/api/auth/me');
  }

  // Maintainer methods
  async getMaintainerOverview() {
    return this.request('/api/maintainer/overview');
  }

  async getOrganizations() {
    return this.request('/api/maintainer/organizations');
  }

  async createOrganization(data: { name: string; slug: string; description?: string }) {
    return this.request('/api/maintainer/organizations', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getUsers() {
    return this.request('/api/maintainer/users');
  }

  async createMaintainer(data: { email: string; name: string; password: string }) {
    return this.request('/api/maintainer/users/maintainer', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getAuditLogs(params?: { page?: number; limit?: number; action?: string; resource?: string }) {
    const searchParams = new URLSearchParams();
    if (params?.page) searchParams.set('page', params.page.toString());
    if (params?.limit) searchParams.set('limit', params.limit.toString());
    if (params?.action) searchParams.set('action', params.action);
    if (params?.resource) searchParams.set('resource', params.resource);
    
    const query = searchParams.toString();
    return this.request(`/api/maintainer/logs/audit${query ? `?${query}` : ''}`);
  }

  async getSystemStatus() {
    return this.request('/api/maintainer/system/status');
  }

  async impersonateUser(userId: string) {
    return this.request(`/api/maintainer/impersonate/${userId}`, {
      method: 'POST',
    });
  }
}

export const api = new ApiClient(API_BASE_URL);
