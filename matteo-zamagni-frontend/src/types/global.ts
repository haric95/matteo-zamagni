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
};

export type HomepageData = {
  items: HomepageItem[];
};

export enum AboutMode {
  BIO = "Bio",
  AWARDS = "Awards",
  RESIDENCIES = "Residencies",
  PERFORMANCES = "Performances",
  SCREENINGS = "Screenings",
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
  [AboutMode.RESIDENCIES]: StrapiAboutComponent[];
  [AboutMode.PERFORMANCES]: StrapiAboutComponent[];
  [AboutMode.SCREENINGS]: StrapiAboutComponent[];
  [AboutMode.TALKS]: StrapiAboutComponent[];
  CV: StrapiImageResponse | null;
  DigitalSales: { label: string; url: string }[] | null;
  RepresentedBy: { label: string; url: string }[] | null;
};
