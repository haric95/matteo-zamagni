import { DEFAULT_ANIMATE_MODE } from "@/const";
import { useGlobalContext } from "@/state/GlobalStore";
import { AnimatePresence, motion } from "framer-motion";
import React from "react";
import { MdClose } from "react-icons/md";
import { TypeAnimation } from "react-type-animation";

type CreditsViewerProps = {
  handleClose: () => void;
};

export const CreditsViewer: React.FC<CreditsViewerProps> = ({
  handleClose,
}) => {
  const { creditsIsOpen } = useGlobalContext();

  return (
    <AnimatePresence>
      {creditsIsOpen && (
        <motion.div
          {...DEFAULT_ANIMATE_MODE}
          className="w-screen height-dynamic-screen fixed left-0 top-0 z-[999] flex justify-center items-center "
        >
          <div
            className="w-full h-full absolute bg-black opacity-80"
            onClick={handleClose}
          />
          <div className="w-[80%] md:w-[33%] h-fit flex justify-center items-center z-10 relative border-white border-[1px]">
            <button
              className="absolute top-[8px] right-[8px] w-fit h-fit flex justify-center icon-hover-glow transition-all duration-500"
              onClick={() => handleClose()}
            >
              <MdClose />
            </button>
            <div className="w-full h-full bg-black p-8 flex flex-col justify-between">
              <div className="mb-8">
                <a href="https://instagram.com/nufolklore" target="_blank">
                  <TypeAnimation
                    sequence={["Site designed by NUFOLKLORE"]}
                    wrapper="span"
                    speed={50}
                    style={{ display: "inline-block" }}
                    className=""
                  />
                </a>
              </div>
              <div>
                <a href={"mailto:harichauhan@protonmail.com"} target="_blank">
                  <TypeAnimation
                    sequence={["Web development by Hari Chauhan"]}
                    wrapper="span"
                    speed={50}
                    style={{ display: "inline-block" }}
                    className=""
                  />
                </a>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
