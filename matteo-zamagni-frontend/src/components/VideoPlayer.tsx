import { useIsMobile } from "@/hooks/useIsMobile";
import React from "react";
import { MdClose } from "react-icons/md";
import ReactPlayer from "react-player";

type VideoPlayerProps = {
  url: string;
  handleClose: () => void;
};

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  url,
  handleClose,
}) => {
  const isMobile = useIsMobile();
  return (
    <div className="w-screen h-[calc(100dvh)] fixed left-0 top-0 z-10 flex justify-center items-center">
      <div
        className="w-full h-full absolute bg-black opacity-80"
        onClick={handleClose}
      />
      <div
        className="w-full md:w-fit md:min-w-[90%] h-full md:h-[90%] flex justify-center items-center z-10 relative"
        onClick={() => {
          handleClose();
        }}
      >
        <button
          className="absolute translate-y-[8px] top-0 right-0 w-8 h-8 flex justify-center icon-hover-glow transition-all duration-500"
          onClick={() => handleClose()}
        >
          <MdClose stroke="white" fill="white" />
        </button>
        <ReactPlayer
          url={url}
          controls
          playing={true}
          width={isMobile ? "100%" : "90%"}
          height={isMobile ? "50%" : "100%"}
        />
      </div>
    </div>
  );
};
