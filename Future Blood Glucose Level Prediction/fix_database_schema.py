import sqlite3
import os

print("=" * 60)
print("🔧 FIXING DATABASE SCHEMA")
print("=" * 60)

db_path = os.path.join('database', 'patient_history.db')

if not os.path.exists(db_path):
    print("❌ Database not found at:", db_path)
    exit(1)

print(f"✅ Database found at: {db_path}")

# Connect to database
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Check current schema
c.execute("PRAGMA table_info(predictions)")
columns = c.fetchall()

print("\n📊 Current columns in 'predictions' table:")
for col in columns:
    print(f"   - {col[1]} ({col[2]})")

# Check if blockchain_tx column exists
has_blockchain_tx = any(col[1] == 'blockchain_tx' for col in columns)

if not has_blockchain_tx:
    print("\n➕ Adding 'blockchain_tx' column...")
    try:
        c.execute("ALTER TABLE predictions ADD COLUMN blockchain_tx TEXT")
        print("✅ Successfully added blockchain_tx column!")
    except Exception as e:
        print(f"❌ Error adding column: {e}")
else:
    print("\n✅ 'blockchain_tx' column already exists")

# Verify the column was added
c.execute("PRAGMA table_info(predictions)")
updated_columns = c.fetchall()
print("\n📊 Updated columns:")
for col in updated_columns:
    mark = "✓" if col[1] == 'blockchain_tx' else " "
    print(f"   {mark} {col[1]} ({col[2]})")

# Commit and close
conn.commit()
conn.close()

print("\n" + "=" * 60)
print("✅ Database fix complete!")
print("=" * 60)