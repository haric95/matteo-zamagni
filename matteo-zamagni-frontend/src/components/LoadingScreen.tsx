import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import React, { PropsWithChildren, useEffect, useState } from "react";

export const LoadingScreen = ({ children }: PropsWithChildren) => {
  return <>{children}</>;
  const { hasLoaded } = useGlobalContext();
  const dispatch = useGlobalContextDispatch();
  const [videoEnded, setVideoEnded] = useState(false);
  const [loadingScreenVisible, setLoadingScreenVisible] = useState(!hasLoaded);

  const handleVideoEnded = () => {
    setVideoEnded(true);
    if (dispatch) {
      setTimeout(() => {
        handleLoaded();
      }, 500);
    }
  };

  const handleLoaded = () => {
    if (dispatch) {
      dispatch({ type: "SET_LOADED" });
    }
  };

  useEffect(() => {
    if (hasLoaded) {
      setTimeout(() => {
        setLoadingScreenVisible(false);
      }, 600);
    }
  }, [hasLoaded]);

  return (
    <>
      {children}
      {loadingScreenVisible && (
        <div
          className={`fixed top-0 left-0 w-screen h-screen loading-screen z-[1000] flex items-center justify-center  transition-all duration-500 ${
            !videoEnded
              ? "bg-background_Dark"
              : "dark:bg-background_Dark bg-background_Light"
          } ${hasLoaded ? "opacity-0" : "opacity-100"}`}
        >
          {!videoEnded && (
            <div className="w-[480px] h-[480px] relative">
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
      )}
    </>
  );
};
