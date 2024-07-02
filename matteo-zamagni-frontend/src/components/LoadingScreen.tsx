import { useLoadInitial } from "@/hooks/useLoadInitial";
import { useGlobalContext } from "@/state/GlobalStore";
import { PropsWithChildren, useEffect, useState } from "react";

export const LoadingScreen = ({ children }: PropsWithChildren) => {
  useLoadInitial();
  const { hasLoaded } = useGlobalContext();
  const [videoEnded, setVideoEnded] = useState(false);
  const [loadingScreenVisible, setLoadingScreenVisible] = useState(true);

  const handleVideoEnded = () => {
    setVideoEnded(true);
  };

  useEffect(() => {
    if (hasLoaded && videoEnded) {
      setLoadingScreenVisible(false);
    }
  }, [hasLoaded, videoEnded]);

  return (
    <>
      {children}
      <div
        className={`fixed top-0 left-0 w-screen h-screen loading-screen z-[1000] flex items-center justify-center  transition-all duration-500 ${
          !videoEnded
            ? "bg-background_Dark"
            : "dark:bg-background_Dark bg-background_Light"
        } ${
          loadingScreenVisible ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      >
        {!videoEnded && (
          <div className="w-[240px] h-[240px] relative">
            <video
              className="absolute w-full h-full"
              src={"/loading-anim.mp4"}
              autoPlay
              muted
              onEnded={handleVideoEnded}
            />
            <div className="absolute w-full h-full vignette" />
          </div>
        )}
      </div>
    </>
  );
};
