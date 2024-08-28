'use client'

import { useState, FormEvent } from 'react'

interface SearchResult {
  'Clinic Notes'?: string
  'Similarity Score'?: number
  error?: string
}

export default function Home() {
  const [query, setQuery] = useState<string>('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [topN, setTopN] = useState<number>(5)
  const [modelType, setModelType] = useState<string>('word2vec')
  const [loading, setLoading] = useState<boolean>(false)

  const handleSearch = async (e: FormEvent) => {
    e.preventDefault()
    setLoading(true)

    try {
      const response = await fetch(`http://localhost:8000/search?query=${encodeURIComponent(query)}&model_type=${modelType}&top_n=${topN}`)
      const data = await response.json()

      if (!response.ok || data.error) {
        throw new Error(data.error || 'Failed to fetch data')
      }

      setResults(data.results)
    } catch (error: any) {
      console.error('Error fetching data:', error.message)
      setResults([{ error: error.message }])
    } finally {
      setLoading(false)
    }
  }

  const handleLoadModel = async () => {
    setLoading(true)
    try {
      const response = await fetch(`http://localhost:8000/load-model?model_type=${modelType}`, {
        method: 'POST',
      })
      const data = await response.json()
      console.log('Model load response:', data)
      alert(`Model ${modelType} loaded successfully!`)
    } catch (error) {
      console.error('Error loading model:', error)
      alert('Failed to load model')
    } finally {
      setLoading(false)
    }
  }

  // Highlight all occurrences of the query words in the text
  const highlightText = (text: string, query: string) => {
    if (!query) return text

    const queryWords = query.split(/\s+/).filter(Boolean) // Split query into words
    const regex = new RegExp(`(${queryWords.join('|')})`, 'gi') // Create a regex for all query words

    return text.split(regex).map((part, i) =>
      regex.test(part) ? (
        <span key={i} className="text-yellow-500 ">{part}</span>
      ) : (
        part
      )
    )
  }

  return (
    <main className="flex min-h-screen flex-col items-center gap-10 p-24">
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex">
        <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
          Search Engine for Clinic Notes&nbsp;
          <code className="font-mono font-bold">app/page.tsx</code>
        </p>
      </div>

      <div className="w-full px-32">
        <form className="w-8/12 mx-auto mb-4" onSubmit={handleSearch}>
          <label className="mb-2 text-sm font-medium text-gray-900 sr-only dark:text-white">Search</label>
          <div className="relative mb-4">
            <div className="absolute inset-y-0 start-0 flex items-center ps-3 pointer-events-none">
              <svg className="w-4 h-4 text-gray-500 dark:text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 20">
                <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="m19 19-4-4m0-7A7 7 0 1 1 1 8a7 7 0 0 1 14 0Z" />
              </svg>
            </div>
            <input
              type="search"
              id="default-search"
              className="block w-full p-4 ps-10 text-sm text-gray-900 border border-gray-300 rounded-lg bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white"
              placeholder="Search Clinic Notes"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              required
            />
          </div>

          <div className="flex gap-4 mb-4">
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="p-2 text-sm border rounded-lg bg-gray-50 dark:bg-gray-700 dark:text-white"
            >
              <option value="word2vec">Word2Vec</option>
              <option value="fasttext">FastText</option>
              <option value="tfidf">TF-IDF</option>
            </select>

            <input
              type="number"
              value={topN}
              onChange={(e) => setTopN(parseInt(e.target.value))}
              className="block p-2 text-sm text-gray-900 border border-gray-300 rounded-lg bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white"
              placeholder="Top N Results"
              min="1"
            />

            <button
              type="button"
              onClick={handleLoadModel}
              className="p-2 text-white bg-green-500 hover:bg-green-600 rounded-lg"
            >
              {loading ? 'Loading...' : 'Load Model'}
            </button>
          </div>

          <button
            type="submit"
            className="w-full p-2 text-white bg-blue-700 hover:bg-blue-800 rounded-lg"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>
      </div>

      <div className="w-7/12 mt-10">
        {results.map((result, index) => (
          <div key={index} className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-lg mb-4">
            {result.error ? (
              <div>
                <h2 className="text-xl font-semibold text-red-600 dark:text-red-400">Error</h2>
                <p className="text-gray-600 dark:text-gray-400">{result.error}</p>
              </div>
            ) : (
              <div className="space-y-2">
                <h5 className="font-semibold text-gray-900 dark:text-white">Result {index + 1}</h5>
                <p className="text-gray-600 dark:text-gray-400">
                  Clinic Notes: {highlightText(result['Clinic Notes'] || '', query)}
                </p>
                <p className="text-gray-600 dark:text-green-400">Similarity Score: {result['Similarity Score']}</p>
              </div>
            )}
          </div>
        ))}
      </div>
    </main>
  )
}
