import { UserButton } from "@clerk/nextjs";
import SearchInterface from "./SearchInterface";

export default function Home() {
  return (
    <main className="search-page">
      <div className="user-menu">
        <UserButton />
      </div>
      <SearchInterface />
    </main>
  );
}
