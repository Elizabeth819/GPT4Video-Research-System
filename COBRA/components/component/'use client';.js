'use client';
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { CardTitle, CardDescription, CardHeader, CardContent, Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { DropdownMenuTrigger, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuItem, DropdownMenuContent, DropdownMenu } from "@/components/ui/dropdown-menu"
import { ScrollArea } from "@/components/ui/scroll-area"
import { CollapsibleTrigger, CollapsibleContent, Collapsible } from "@/components/ui/collapsible"
import { DialogTrigger, DialogTitle, DialogHeader, DialogContent, Dialog } from "@/components/ui/dialog"
import { useChat } from 'ai/react'
import axios from "axios"
import { useState, useRef, useEffect, useMemo } from "react"
import ReactPlayer from 'react-player';
import actionSummary from '../../app/data/desktop-monitoring-task-tsmc2.json'
import Image from "next/image";

// Global function to parse ISO 8601 duration to seconds
const parseDuration = (duration) => {
  const timeParts = duration.match(/PT(\d+H)?(\d+M)?(\d+(\.\d+)?S)?/);
  const hours = (parseFloat(timeParts[1]) || 0) * 3600;
  const minutes = (parseFloat(timeParts[2]) || 0) * 60;
  const seconds = parseFloat(timeParts[3]) || 0;
  return hours + minutes + seconds;
};

// Convert offset (ISO 8601) to seconds
const parseOffset = (offset) => {
  return parseDuration(offset);
};

// Video Search Results Component
const VideoSearch = ({ res }) => {
  const startFrame = parseOffset(res.offset);
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <span className="text-sm line-clamp-1">{res.offset}</span>
        <span className="text-sm line-clamp-1">{res.duration}</span>
        <span className="text-sm line-clamp-1">{res.properties.description.substring(0, 40)}</span>
      </div>
      <PlayIcon className="w-4 h-4" onClick={() => handleSearchResultClick(startFrame)} />
    </div>
  );
};

// Video Summary Display Component
const VideoSummaryDisplay = ({ currentSecond }) => {
  const scrollRef = useRef(null);

  const currentSummaries = useMemo(() => {
    return actionSummary.map((summary) => {
      const start = parseOffset(summary.offset);
      const duration = parseDuration(summary.duration);
      const end = start + duration;
      const isCurrent = currentSecond >= start && currentSecond <= end;
      return { ...summary, isCurrent };
    });
  }, [currentSecond]);

  useEffect(() => {
    const currentSummaryElement = scrollRef.current.querySelector('.current-summary');
    if (currentSummaryElement) {
      currentSummaryElement.scrollIntoView({ behavior: 'auto', block: 'nearest' });
    }
  }, [currentSummaries]);

  return (
    <ScrollArea className="h-96 w-full" ref={scrollRef}>
      {currentSummaries.map((summary, index) => (
        <div
          key={index}
          className={`w-full ${summary.isCurrent ? 'current-summary' : ''}`}
        >
          <p className="font-bold">{summary.offset} - {summary.duration}</p>
          <p>{summary.properties.description}</p>
          <p><strong>Sensitive Action:</strong> {summary.properties.sensitiveActions[0]?.description || 'N/A'}</p>
        </div>
      ))}
    </ScrollArea>
  );
};

// Action Card Display Component
const ActionCard = ({ actionData, currentSecond }) => {
  const scrollRef = useRef(null);

  const currentActions = useMemo(() => {
    return actionData.map((action) => {
      const start = parseOffset(action.offset);
      const duration = parseDuration(action.duration);
      const end = start + duration;
      const isCurrent = currentSecond >= start && currentSecond <= end;
      return { ...action, isCurrent };
    });
  }, [currentSecond, actionData]);

  useEffect(() => {
    const currentActionElement = scrollRef.current.querySelector('.current-action');
    if (currentActionElement) {
      currentActionElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [currentActions]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Current Action Analysis</CardTitle>
      </CardHeader>
      <CardContent className="h-96 overflow-y-auto" ref={scrollRef}>
        {currentActions.map((action, index) => (
          <div key={index} className={`w-full ${action.isCurrent ? 'current-action' : ''}`}>
            <Collapsible className="space-y-2">
              <div className="flex items-center justify-between space-x-4 px-4">
                <h4 className="text-sm font-semibold">Action {index + 1}</h4>
                <CollapsibleTrigger asChild>
                  <Button size="sm" variant="ghost">
                    T<span className="sr-only">Toggle</span>
                  </Button>
                </CollapsibleTrigger>
              </div>
              <CollapsibleContent className="space-y-2 px-4">
                <p className="text-sm">{action.properties.description}</p>
                <p className="text-sm">
                  <strong>Risk Level:</strong> {action.properties.sensitiveActions[0]?.['risk Level'] || 'N/A'}
                </p>
                <p className="text-sm">
                  <strong>Sensitive Action:</strong> {action.properties.sensitiveActions[0]?.description || 'N/A'}
                </p>
              </CollapsibleContent>
            </Collapsible>
          </div>
        ))}
      </CardContent>
    </Card>
  );
};

// Main Player Component
export function Player(props) {
  const [played, setPlayed] = useState(0);
  const [playing, setPlaying] = useState(true);
  const player = useRef(null);
  const [currentSecond, setCurrentSecond] = useState(0);
  const [searchResults, setSearchResults] = useState([]);
  const [searchText, setSearchText] = useState('');

  const videoSummaries = actionSummary;
  const [videoURL, setVideoURL] = useState('Video recording for GPT-4O detection-20240711_133916-Meeting Recording.mp4');

  const handleProgress = ({ playedSeconds }) => {
    setCurrentSecond(playedSeconds);
  };

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    axios.post('/api/cog', { messages: searchText })
      .then((response) => {
        let count = 0;
        const results = response.data.message.map(item => ({
          ...item,
          offset: parseOffset(item.offset),
          duration: parseDuration(item.duration),
          rank: ++count
        }));
        setSearchResults(results);
      })
      .catch((error) => {
        console.error(error);
      });
  };

  return (
    <div className="grid min-h-screen w-full lg:grid-cols-[280px_1fr]">
      <div className="hidden border-r bg-gray-100/40 lg:block dark:bg-gray-800/40">
        <div className="flex h-full max-h-screen flex-col gap-2">
          <div className="flex h-[60px] items-center border-b px-6">
            <Link className="flex items-center gap-2 font-semibold" href="#">
              <span>COBRA</span>
            </Link>
          </div>
          <div className="flex-1 overflow-auto py-2">
            <nav className="grid items-start px-4 text-sm font-medium">
              <Link className="group flex items-center gap-3 rounded-lg px-3 py-2 text-gray-500 transition-all hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50" href="#">
                Recent Videos
              </Link>
              <Link className="group flex items-center gap-3 rounded-lg px-3 py-2 text-gray-500 transition-all hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50" href="#">
                Search
              </Link>
              <Link className="group flex items-center gap-3 rounded-lg px-3 py-2 text-gray-500 transition-all hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50" href="#">
                Upload Video
              </Link>
            </nav>
          </div>
        </div>
      </div>

      <div className="flex flex-col">
        <header className="flex h-14 lg:h-[60px] items-center gap-4 border-b bg-gray-100/40 px-6 dark:bg-gray-800/40">
          <div className="w-full flex-1">
            <form>
              <div className="relative">
                <Input className="w-full bg-white shadow-none appearance-none pl-8 md:w-2/3 lg:w-1/3 dark:bg-gray-950" placeholder="Search video..." type="search" onChange={(e) => setSearchText(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSearchSubmit(e)} />
              </div>
            </form>
          </div>
        </header>

        <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-6">
          <div className="flex w-full gap-4">
            <div className="w-full rounded-lg overflow-hidden">
              <ReactPlayer ref={player} url={videoURL} playing={playing} controls={true} onProgress={handleProgress} className="w-full aspect-video rounded-md bg-gray-100 dark:bg-gray-800" />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 h-auto">
            <Card>
              <CardHeader>
                <CardTitle>Matched Searches</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-2">
                {searchResults.map((result, index) => (
                  <VideoSearch key={index} res={result} />
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Action Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <VideoSummaryDisplay currentSecond={currentSecond} />
              </CardContent>
            </Card>

            <ActionCard actionData={videoSummaries} currentSecond={currentSecond} />
          </div>
        </main>
      </div>
    </div>
  );
}
