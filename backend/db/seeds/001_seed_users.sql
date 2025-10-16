-- Seed some sample users
INSERT INTO users (email, name) VALUES
    ('john.doe@example.com', 'John Doe'),
    ('jane.smith@example.com', 'Jane Smith'),
    ('bob.johnson@example.com', 'Bob Johnson')
ON CONFLICT (email) DO NOTHING;
