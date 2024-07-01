// Import Swiper React components
import Image from "next/image";
import { MdChevronLeft, MdChevronRight, MdClose } from "react-icons/md";
import { Swiper as SwiperCore } from "swiper/core";
import { Pagination } from "swiper/modules";
import { Swiper, SwiperSlide } from "swiper/react";

// Import Swiper styles
import { useEffect, useState } from "react";
import "swiper/css";
import "swiper/css/pagination";
import { useIsMobile } from "@/hooks/useIsMobile";

type ImageGalleryProps = {
  images: { imageURL: string; alt: string }[];
  initialSlide: number;
  handleClose: () => void;
};

export const ImageGallery: React.FC<ImageGalleryProps> = ({
  images,
  initialSlide,
  handleClose,
}) => {
  const [swiper, setSwiper] = useState<SwiperCore | null>(null);
  const [currentSlide, setCurrentSlide] = useState(initialSlide);
  const isMobile = useIsMobile();

  const handleClick = (dir: "left" | "right") => {
    if (swiper) {
      if (dir === "left") {
        setCurrentSlide((old) => (old + images.length - 1) % images.length);
      } else {
        setCurrentSlide((old) => (old + 1) % images.length);
      }
    }
  };

  useEffect(() => {
    if (swiper) {
      swiper.slideTo(currentSlide);
    }
  }, [swiper, currentSlide]);

  return (
    <div className="w-screen h-screen fixed left-0 top-0 z-10 flex justify-center items-center">
      <div
        className="w-full h-full absolute bg-black opacity-90"
        onClick={handleClose}
      />
      <div className="w-full h-1/2 md:w-3/4 md:h-3/4 flex justify-center items-center z-10 relative">
        <button
          className="absolute top-0 right-0 w-8 h-8 flex justify-center icon-hover-glow hover:scale-105 transition-all duration-500"
          onClick={() => handleClose()}
        >
          <MdClose />
        </button>
        <button
          className="w-[5%] h-1/2 flex items-center justify-center"
          onClick={() => handleClick("left")}
        >
          <div className="h-fit w-fit icon-hover-glow hover:scale-110 transition-all duration-500">
            <MdChevronLeft width={16} height={16} />
          </div>
        </button>
        <Swiper
          // install Swiper modules
          modules={[Pagination]}
          onSwiper={(swiper) => setSwiper(swiper)}
          // modules={[Pagination, Navigation]}
          spaceBetween={0}
          slidesPerView={1}
          initialSlide={initialSlide}
          navigation
          pagination={{ clickable: true }}
          className="w-full h-full"
          style={{
            // @ts-ignore
            "--swiper-pagination-color": "#FEF781",
            "--swiper-pagination-bullet-inactive-color": "#ffffff",
            "--swiper-pagination-bullet-inactive-opacity": "1",
          }}
        >
          {images.map((image, index) => {
            return (
              <SwiperSlide key={index} className="w-full h-full">
                <Image
                  src={image.imageURL}
                  alt={image.alt}
                  layout="fill"
                  objectFit={isMobile ? "contain" : "cover"}
                />
              </SwiperSlide>
            );
          })}
        </Swiper>
        <button
          className="w-[5%] h-1/2 flex items-center justify-center z-10"
          onClick={() => handleClick("right")}
        >
          <div className="h-fit w-fit icon-hover-glow hover:scale-110 transition-all duration-500">
            <MdChevronRight width={"32px"} height={"32px"} />
          </div>
        </button>
      </div>
    </div>
  );
};
