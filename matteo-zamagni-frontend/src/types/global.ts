import {
  HorizontalLines,
  Circle,
  Star,
  BackChevrons,
} from "@/components/Icons";
import { StrapiImageResponse } from "@/hooks/useStrapi";

export type Dim2D = { x: number; y: number };
export type Pos2D = { x: number; y: number };
export type PolarPos2d = { radius: number; theta: number }; // Theta in radians
export type PosAndDim2D = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type Grid = boolean[][];

export enum HomepageItemType {
  EXHIBITION = "Exhibition",
  PROJECT = "Project",
}

export type HomepageItem = {
  position: { x: number; y: number };
  type: HomepageItemType;
  title: string;
  year: string;
  image: StrapiImageResponse;
  slug: string;
  tags: string;
  iconScale?: number;
};

export type HomepageData = {
  items: HomepageItem[];
};

export enum AboutMode {
  BIO = "Bio",
  AWARDS = "Awards",
  EXHIBITIONS = "Exhibitions",
  TALKS = "Talks",
}

export enum StrapiAboutComponentType {
  Title = "about.about-title",
  Year = "about.about-year",
  Item = "about.about-item",
  Text = "about.about-text",
}

type StrapiTitleComponent = {
  __component: StrapiAboutComponentType.Title;
  Title: string;
};

type StrapiYearComponent = {
  __component: StrapiAboutComponentType.Year;
  Year: string;
};

type StrapiItemComponent = {
  __component: StrapiAboutComponentType.Item;
  Label: string;
  Name: string;
  Details: string;
};

type StrapiTextComponent = {
  __component: StrapiAboutComponentType.Text;
  Text: string;
};

export type StrapiAboutComponent =
  | StrapiTitleComponent
  | StrapiYearComponent
  | StrapiItemComponent
  | StrapiTextComponent;

export type AboutPageData = {
  [AboutMode.BIO]: StrapiAboutComponent[];
  [AboutMode.AWARDS]: StrapiAboutComponent[];
  [AboutMode.EXHIBITIONS]: StrapiAboutComponent[];
  [AboutMode.TALKS]: StrapiAboutComponent[];
  CV: StrapiImageResponse | null;
  DigitalSales: { label: string; url: string }[] | null;
  RepresentedBy: { label: string; url: string }[] | null;
};

export enum WorkIndexType {
  INSTALLATION = "Installation",
  PERFORMANCE = "Performance",
  FILM = "Film",
  PRINT = "Print",
}

export const WORK_INDEX_TYPE_ARRAY = [
  WorkIndexType.INSTALLATION,
  WorkIndexType.PERFORMANCE,
  WorkIndexType.FILM,
  WorkIndexType.PRINT,
];

export const WorkIndexTypeIcon = {
  [WorkIndexType.INSTALLATION]: HorizontalLines,
  [WorkIndexType.PERFORMANCE]: Circle,
  [WorkIndexType.FILM]: Star,
  [WorkIndexType.PRINT]: BackChevrons,
};

export type IndexItem = {
  type: HomepageItemType;
  title: string;
  year: string;
  slug: string;
  tags: string;
};

export type IndexPageData = {
  items: IndexItem[];
  PixelBackgroundAnimation: StrapiImageResponse;
};
