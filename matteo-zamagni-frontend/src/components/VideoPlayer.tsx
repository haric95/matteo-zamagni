import React from "react";
import { MdClose } from "react-icons/md";
import ReactPlayer from "react-player/vimeo";

type VideoPlayerProps = {
  url: string;
  handleClose: () => void;
};

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  url,
  handleClose,
}) => {
  return (
    <div className="w-screen h-screen fixed left-0 top-0 z-10 flex justify-center items-center">
      <div
        className="w-full h-full absolute bg-black opacity-90"
        onClick={handleClose}
      />
      <div className="w-[90%] h-[90%] flex justify-center items-center z-10 relative">
        <button
          className="absolute top-0 right-0 w-8 h-8 flex justify-center icon-hover-glow transition-all duration-500"
          onClick={() => handleClose()}
        >
          <MdClose />
        </button>
        <ReactPlayer
          url={url}
          controls
          playing={true}
          width={"90%"}
          height={"90%"}
        />
      </div>
    </div>
  );
};
