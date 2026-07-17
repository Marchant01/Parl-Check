import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/')({
  component: Index,
})

const page_content = (
  <p className='text-3xl text-center'>
    Parl-Check är en webbtjänst med syfte att ge insikt i uttalanden och beslut som tas i den svenska riksdagen. 
    Med hjälp av data från från riksdagens öppna API som hanteras med RAG och en LLM, kan användare ställa frågor om riksdagen och få svar baserat på faktiska data.
    
    {/* Explain the data here */}
  </p>
);

function Index() {
  return (
    <p className="text-3xl text-center">
      {page_content}
    </p>
  )
}