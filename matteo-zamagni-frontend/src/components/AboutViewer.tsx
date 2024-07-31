import { StrapiAboutComponent, StrapiAboutComponentType } from "@/types/global";
import React from "react";
import Markdown from "react-markdown";

type AboutViewerProps = {
  content: StrapiAboutComponent[] | null;
};

export const AboutViewer: React.FC<AboutViewerProps> = ({ content }) => {
  return content
    ? content.map((item) => {
        switch (item.__component) {
          case StrapiAboutComponentType.Title:
            return (
              <h1 className="text-xl mb-4">
                <b>{item.Title}</b>
              </h1>
            );
          case StrapiAboutComponentType.Year:
            return (
              <div className="flex justify-end">
                <p className="text-lg text-right w-full md:w-2/3 border-b-[1px] border-black py-2">
                  {item.Year}
                </p>
              </div>
            );
          case StrapiAboutComponentType.Item:
            return (
              <div className="flex-wrap md:flex-nowrap flex mt-2">
                <div className="w-full md:w-1/3 mb-4 md:mb-0 font-bold md:font-normal">
                  <p>{item.Label}</p>
                </div>
                <div className="w-full md:w-2/3 flex">
                  <div className="w-1/2 border-b-[1px] border-black pb-2">
                    <p>{item.Name}</p>
                  </div>
                  <div className="w-1/2 border-b-[1px] border-black pb-2">
                    <p>{item.Details}</p>
                  </div>
                </div>
              </div>
            );
          case StrapiAboutComponentType.Text:
            return (
              <div className="py-2">
                <Markdown>{item.Text}</Markdown>
              </div>
            );
          default:
            return null;
        }
      })
    : null;
};
