import { Client, Account, Databases, ID } from "appwrite";

// When NEXT_PUBLIC_DEV_API_KEY is set we run fully offline and never
// touch Appwrite. Fall back to harmless placeholders so the SDK init
// doesn't crash if the env vars are empty.
const endpoint =
  process.env.NEXT_PUBLIC_APPWRITE_ENDPOINT || "https://cloud.appwrite.io/v1";
const projectId = process.env.NEXT_PUBLIC_APPWRITE_PROJECT_ID || "dev-placeholder";

const client = new Client().setEndpoint(endpoint).setProject(projectId);

export const account = new Account(client);
export const databases = new Databases(client);
export { client, ID };
