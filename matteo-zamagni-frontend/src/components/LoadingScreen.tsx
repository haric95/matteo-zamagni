import { useLoadInitial } from "@/hooks/useLoadInitial";
import { useGlobalContext } from "@/state/GlobalStore";
import { PropsWithChildren, useEffect, useState } from "react";
import { Loader } from "./Loader";

export const LoadingScreen = ({ children }: PropsWithChildren) => {
  useLoadInitial();
  const { hasLoaded } = useGlobalContext();
  const [videoEnded, setVideoEnded] = useState(false);
  const [loadingScreenVisible, setLoadingScreenVisible] = useState(true);
  const [timeoutState, setTimeoutState] = useState<number | null>(null);

  const handleVideoEnded = () => {
    setVideoEnded(true);
  };

  useEffect(() => {
    if (hasLoaded && videoEnded) {
      setLoadingScreenVisible(false);
    }
  }, [hasLoaded, videoEnded]);

  // Handle case where video doesn't autoplay, by creating 5s timeout
  useEffect(() => {
    if (!videoEnded && !timeoutState) {
      const timeout = window.setTimeout(() => {
        setVideoEnded(true);
        setTimeoutState(null);
      }, 5000);
      setTimeoutState(timeout);
    }
  }, [timeoutState, videoEnded]);

  return (
    <>
      {children}
      <div
        className={`fixed top-0 left-0 w-screen height-dynamic-screen loading-screen z-[1000] flex items-center justify-center  transition-all duration-500 ${
          !videoEnded
            ? "bg-background_Dark"
            : "dark:bg-background_Dark bg-background_Light"
        } ${
          loadingScreenVisible ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      >
        {!videoEnded && (
          <div className="w-[240px] h-[240px] relative fade-in">
            <div
              className="w-[160px] h-[160px] absolute left-[50%] top-[50%]"
              style={{ transform: "translate(-50%, -50%)" }}
            />
            {/* <Loader /> */}
            <img src="/loader.gif" alt="loader" className="w-full h-full" />
          </div>
        )}
      </div>
    </>
  );
};
