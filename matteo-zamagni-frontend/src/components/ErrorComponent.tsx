"use client";

import { DEFAULT_ANIMATE_MODE } from "@/const";
import { AnimatePresence, motion } from "framer-motion";
import React from "react";
import { TypeAnimation } from "react-type-animation";

export const ErrorComponent: React.FC = () => {
  return (
    <AnimatePresence>
      {
        <motion.div
          {...DEFAULT_ANIMATE_MODE}
          className="w-screen h-screen fixed left-0 top-0 z-[999] flex justify-center items-center "
        >
          <div className="w-full h-full absolute bg-black opacity-80" />
          <div className="w-[80%] md:w-[33%] h-fit flex justify-center items-center z-10 relative border-white border-[1px]">
            <div className="w-full h-full bg-black p-8 flex flex-col justify-between">
              <div>
                <a href={"mailto:harichauhan@protonmail.com"} target="_blank">
                  <TypeAnimation
                    sequence={[
                      "There has been an error. Please refresh. \n\nIf the error persists, please try again later.",
                    ]}
                    wrapper="span"
                    speed={50}
                    style={{ display: "inline-block" }}
                    className="whitespace-pre-line"
                  />
                </a>
              </div>
            </div>
          </div>
        </motion.div>
      }
    </AnimatePresence>
  );
};
