import {
  StrapiAboutComponent,
  StrapiAboutComponentType,
} from "@/app/about/page";
import React from "react";
import Markdown from "react-markdown";

type AboutViewerProps = {
  content: StrapiAboutComponent[] | null;
};

export const AboutViewer: React.FC<AboutViewerProps> = ({ content }) => {
  return content
    ? content.map((item) => {
        if (item.__component === StrapiAboutComponentType.Text) {
          console.log("here");
        }
        switch (item.__component) {
          case StrapiAboutComponentType.Title:
            return (
              <h1 className="text-right text-xl mb-4">
                <b>{item.Title}</b>
              </h1>
            );
          case StrapiAboutComponentType.Year:
            return (
              <div className="flex justify-end">
                <p className="text-right w-2/3 border-b-[1px] border-black py-2">
                  {item.Year}
                </p>
              </div>
            );
          case StrapiAboutComponentType.Item:
            return (
              <div className="flex mt-2">
                <div className="w-1/3">
                  <p>{item.Label}</p>
                </div>
                <div className="w-2/3 flex">
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
            break;
          default:
            return null;
        }
      })
    : null;
};
