import ResultsDashboard from "@/components/results/ResultsDashboard";

export default async function ResultsPage({
  params,
}: {
  params: Promise<{ session_id: string }>;
}) {
  const { session_id } = await params;
  return <ResultsDashboard sessionId={session_id} />;
}
