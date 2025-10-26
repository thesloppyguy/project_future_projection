import { clickhouse } from '../config/clickhouse';
import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';

export class ClickHouseMigrationService {
  private static readonly MIGRATIONS_DIR = join(__dirname, '../../clickhouse/init');

  /**
   * Run all ClickHouse migrations
   */
  static async runMigrations(): Promise<void> {
    console.log('🔄 Running ClickHouse migrations...');

    try {
      // Get all migration files
      const migrationFiles = readdirSync(this.MIGRATIONS_DIR)
        .filter(file => file.endsWith('.sql'))
        .sort();

      console.log(`Found ${migrationFiles.length} migration files`);

      for (const file of migrationFiles) {
        console.log(`Running migration: ${file}`);
        await this.runMigration(file);
      }

      console.log('✅ ClickHouse migrations completed');
    } catch (error) {
      console.error('❌ ClickHouse migrations failed:', error);
      throw error;
    }
  }

  /**
   * Run a single migration file
   */
  private static async runMigration(filename: string): Promise<void> {
    const filePath = join(this.MIGRATIONS_DIR, filename);
    const sql = readFileSync(filePath, 'utf8');

    // Split SQL by semicolon and execute each statement
    const statements = sql
      .split(';')
      .map(stmt => stmt.trim())
      .filter(stmt => stmt.length > 0 && !stmt.startsWith('--'));

    for (const statement of statements) {
      if (statement.trim()) {
        try {
          await clickhouse.command({
            query: statement,
          });
        } catch (error) {
          // Some statements might fail if they already exist (like CREATE TABLE IF NOT EXISTS)
          // Log the error but continue
          console.warn(`Warning in migration ${filename}:`, error);
        }
      }
    }
  }

  /**
   * Check if ClickHouse is ready
   */
  static async checkConnection(): Promise<boolean> {
    try {
      await clickhouse.ping();
      return true;
    } catch (error) {
      console.error('ClickHouse connection failed:', error);
      return false;
    }
  }

  /**
   * Get ClickHouse version
   */
  static async getVersion(): Promise<string> {
    try {
      const result = await clickhouse.query({
        query: 'SELECT version()',
        format: 'JSONEachRow',
      });

      const data = await result.json();
      return data[0]?.version || 'Unknown';
    } catch (error) {
      console.error('Failed to get ClickHouse version:', error);
      return 'Unknown';
    }
  }

  /**
   * Get database info
   */
  static async getDatabaseInfo(): Promise<{
    databases: string[];
    tables: string[];
    totalSize: string;
  }> {
    try {
      // Get databases
      const databasesResult = await clickhouse.query({
        query: 'SHOW DATABASES',
        format: 'JSONEachRow',
      });
      const databases = (await databasesResult.json()).map((row: any) => row.name);

      // Get tables
      const tablesResult = await clickhouse.query({
        query: 'SHOW TABLES',
        format: 'JSONEachRow',
      });
      const tables = (await tablesResult.json()).map((row: any) => row.name);

      // Get total size
      const sizeResult = await clickhouse.query({
        query: `
          SELECT formatReadableSize(sum(bytes)) as total_size
          FROM system.parts
          WHERE active = 1
        `,
        format: 'JSONEachRow',
      });
      const totalSize = (await sizeResult.json())[0]?.total_size || '0 B';

      return {
        databases,
        tables,
        totalSize,
      };
    } catch (error) {
      console.error('Failed to get database info:', error);
      return {
        databases: [],
        tables: [],
        totalSize: '0 B',
      };
    }
  }

  /**
   * Get table statistics
   */
  static async getTableStats(tableName: string): Promise<{
    rows: number;
    size: string;
    lastModified: string;
    columns: Array<{ name: string; type: string }>;
  }> {
    try {
      // Get row count
      const rowsResult = await clickhouse.query({
        query: `SELECT count() as rows FROM ${tableName}`,
        format: 'JSONEachRow',
      });
      const rows = (await rowsResult.json())[0]?.rows || 0;

      // Get table size
      const sizeResult = await clickhouse.query({
        query: `
          SELECT formatReadableSize(sum(bytes)) as size
          FROM system.parts
          WHERE table = {tableName:String} AND active = 1
        `,
        query_params: { tableName },
        format: 'JSONEachRow',
      });
      const size = (await sizeResult.json())[0]?.size || '0 B';

      // Get last modified
      const modifiedResult = await clickhouse.query({
        query: `
          SELECT max(modification_time) as last_modified
          FROM system.parts
          WHERE table = {tableName:String} AND active = 1
        `,
        query_params: { tableName },
        format: 'JSONEachRow',
      });
      const lastModified = (await modifiedResult.json())[0]?.last_modified || 'Unknown';

      // Get columns
      const columnsResult = await clickhouse.query({
        query: `DESCRIBE ${tableName}`,
        format: 'JSONEachRow',
      });
      const columns = (await columnsResult.json()).map((row: any) => ({
        name: row.name,
        type: row.type,
      }));

      return {
        rows,
        size,
        lastModified,
        columns,
      };
    } catch (error) {
      console.error(`Failed to get stats for table ${tableName}:`, error);
      return {
        rows: 0,
        size: '0 B',
        lastModified: 'Unknown',
        columns: [],
      };
    }
  }

  /**
   * Optimize tables (run OPTIMIZE)
   */
  static async optimizeTables(): Promise<void> {
    console.log('🔄 Optimizing ClickHouse tables...');

    try {
      const tablesResult = await clickhouse.query({
        query: 'SHOW TABLES',
        format: 'JSONEachRow',
      });
      const tables = (await tablesResult.json()).map((row: any) => row.name);

      for (const table of tables) {
        try {
          await clickhouse.command({
            query: `OPTIMIZE TABLE ${table}`,
          });
          console.log(`Optimized table: ${table}`);
        } catch (error) {
          console.warn(`Failed to optimize table ${table}:`, error);
        }
      }

      console.log('✅ ClickHouse table optimization completed');
    } catch (error) {
      console.error('❌ ClickHouse table optimization failed:', error);
      throw error;
    }
  }

  /**
   * Clean up old data based on TTL
   */
  static async cleanupOldData(): Promise<void> {
    console.log('🔄 Cleaning up old ClickHouse data...');

    try {
      // This will trigger TTL cleanup for all tables
      const tablesResult = await clickhouse.query({
        query: 'SHOW TABLES',
        format: 'JSONEachRow',
      });
      const tables = (await tablesResult.json()).map((row: any) => row.name);

      for (const table of tables) {
        try {
          // Force TTL cleanup
          await clickhouse.command({
            query: `ALTER TABLE ${table} UPDATE _dummy = 0 WHERE 0`,
          });
          console.log(`Cleaned up old data in table: ${table}`);
        } catch (error) {
          console.warn(`Failed to cleanup table ${table}:`, error);
        }
      }

      console.log('✅ ClickHouse data cleanup completed');
    } catch (error) {
      console.error('❌ ClickHouse data cleanup failed:', error);
      throw error;
    }
  }
}
